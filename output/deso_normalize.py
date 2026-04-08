"""DESO AG08+AG09: Schema mapping, field extraction, normalization for tableB."""
import pandas as pd
import re
import html
import os

df = pd.read_csv(r'C:\Users\Ravi\Downloads\tableB (3).csv', encoding='utf-8')
print(f'Input: {len(df)} rows x {len(df.columns)} cols')

# =========================================================================
# STEP 1: Decode HTML entities everywhere
# =========================================================================
for col in ['title', 'authors']:
    df[col] = df[col].apply(lambda x: html.unescape(str(x)) if pd.notna(x) else x)

html_check = df['title'].str.contains(r'&#\d+;', regex=True, na=False).sum()
print(f'After HTML decode - remaining entities in title: {html_check}')

# =========================================================================
# STEP 2: Extract year from title when year column is null
# =========================================================================
def extract_year_from_title(row):
    if pd.notna(row['year']):
        return int(row['year'])
    match = re.findall(r'\b(19\d{2}|200\d)\b', str(row['title']))
    if match:
        return int(match[-1])
    return None

df['year'] = df.apply(extract_year_from_title, axis=1)
df['year'] = df['year'].astype('Int64')

# =========================================================================
# STEP 3: Extract venue from title when venue column is null
# =========================================================================
VENUE_PATTERNS = [
    'international conference on management of data',
    'acm sigmod record',
    'very large data bases',
    'the vldb journal -- the international journal on very large data bases',
    'vldb journal',
    'acm transactions on database systems',
    'acm transactions on database systems ( tods )',
    'acm transactions on information systems',
    'ieee transactions on knowledge and data engineering',
    'ieee data engineering bulletin',
    'data mining and knowledge discovery',
    'knowledge and information systems',
    'distributed and parallel databases',
    'world wide web',
    'journal of computer and system sciences',
    'information systems',
    'sigmod record',
]
VENUE_PATTERNS.sort(key=len, reverse=True)

def extract_venue_from_title(row):
    if pd.notna(row['venue']) and str(row['venue']).strip():
        return row['venue']
    t = str(row['title']).lower()
    for vp in VENUE_PATTERNS:
        if vp in t:
            return vp
    return row['venue']

df['venue'] = df.apply(extract_venue_from_title, axis=1)

# =========================================================================
# STEP 4: Extract authors from title when authors column is null
# =========================================================================
def extract_authors_from_title(row):
    if pd.notna(row['authors']) and str(row['authors']).strip():
        return row['authors']
    t = str(row['title'])
    t_lower = t.lower()
    # Remove venue and year to isolate potential author names
    for vp in VENUE_PATTERNS:
        t_lower = t_lower.replace(vp, '')
    t_lower = re.sub(r'\b(19|20)\d{2}\b', '', t_lower).strip()

    parts = t_lower.split(',')
    if len(parts) >= 2:
        name_parts = []
        for p in parts:
            p = p.strip().rstrip('.')
            words = p.split()
            if 1 <= len(words) <= 5 and all(w.isalpha() or '.' in w for w in words) and len(p) < 50:
                name_parts.append(p)

        if len(name_parts) >= 2:
            first_name = name_parts[0].strip()
            idx = t_lower.find(first_name)
            if idx > 10:
                # Also capture the text segment before the first matched name
                # as it may contain the first author (e.g., "title text firstname lastname , ...")
                prefix = t_lower[:idx].strip()
                prefix_words = prefix.split()
                # Check if the last 2-4 words of prefix look like a person name
                for n in [4, 3, 2]:
                    if len(prefix_words) >= n:
                        candidate = ' '.join(prefix_words[-n:])
                        cand_words = candidate.split()
                        if all(w.isalpha() or '.' in w for w in cand_words) and len(candidate) < 40:
                            # Verify it looks name-like (not title words)
                            # Names typically have 2-4 short words, not common English words
                            name_parts.insert(0, candidate)
                            break
                authors_str = ', '.join(name_parts)
                return authors_str
    return row['authors']

df['authors'] = df.apply(extract_authors_from_title, axis=1)

# =========================================================================
# STEP 5: Clean the title - remove extracted venue, year, authors
# =========================================================================
def clean_title(row):
    t = str(row['title'])
    original_t = t

    if pd.notna(row['venue']):
        venue_lower = str(row['venue']).lower()
        t_lower = t.lower()
        idx = t_lower.find(venue_lower)
        if idx >= 0:
            t = t[:idx] + t[idx + len(venue_lower):]

    if pd.notna(row['year']):
        t = re.sub(r'\b' + str(int(row['year'])) + r'\b', '', t)

    if pd.notna(row['authors']) and str(row['authors']).strip():
        authors_list = [a.strip() for a in str(row['authors']).split(',')]
        for author in authors_list:
            pattern = re.escape(author)
            t = re.sub(pattern, '', t, flags=re.IGNORECASE)

    # Aggressive cleanup of residual punctuation
    t = re.sub(r',\s*,', ',', t)           # double commas
    t = re.sub(r',\s*,', ',', t)           # triple -> double pass
    t = re.sub(r'^\s*[,\s]+', '', t)       # leading commas/spaces
    t = re.sub(r'[,\s]+\s*$', '', t)       # trailing commas/spaces
    t = re.sub(r'\s+', ' ', t).strip()     # multiple spaces
    t = t.strip(' ,.:;-')

    if len(t) < 10 and len(original_t) > 20:
        return original_t
    return t

df['title_clean'] = df.apply(clean_title, axis=1)

# =========================================================================
# STEP 6: Normalize author names
# =========================================================================
def normalize_authors(val):
    if pd.isna(val):
        return val
    authors = [a.strip() for a in str(val).split(',')]
    normalized = []
    for a in authors:
        parts = a.split()
        name_parts = []
        for p in parts:
            if len(p) <= 2 and '.' in p:
                name_parts.append(p.upper())
            elif len(p) == 1:
                name_parts.append(p.upper() + '.')
            else:
                name_parts.append(p.title())
        normalized.append(' '.join(name_parts))
    return ', '.join(normalized)

df['authors_clean'] = df['authors'].apply(normalize_authors)

# =========================================================================
# STEP 7: Normalize venue names to canonical forms
# =========================================================================
VENUE_CANONICAL = {
    'international conference on management of data': 'ACM SIGMOD International Conference on Management of Data',
    'very large data bases': 'International Conference on Very Large Data Bases (VLDB)',
    'acm sigmod record': 'ACM SIGMOD Record',
    'the vldb journal -- the international journal on very large data bases': 'The VLDB Journal',
    'acm transactions on database systems ( tods )': 'ACM Transactions on Database Systems (TODS)',
    'acm transactions on database systems': 'ACM Transactions on Database Systems (TODS)',
    'acm transactions on information systems': 'ACM Transactions on Information Systems (TOIS)',
    'ieee transactions on knowledge and data engineering': 'IEEE Transactions on Knowledge and Data Engineering',
    'ieee data engineering bulletin': 'IEEE Data Engineering Bulletin',
    'data mining and knowledge discovery': 'Data Mining and Knowledge Discovery',
    'knowledge and information systems': 'Knowledge and Information Systems',
    'distributed and parallel databases': 'Distributed and Parallel Databases',
    'world wide web': 'World Wide Web',
    'journal of computer and system sciences': 'Journal of Computer and System Sciences',
    'information systems': 'Information Systems',
    'sigmod record': 'ACM SIGMOD Record',
}

def normalize_venue(val):
    if pd.isna(val):
        return val
    v = str(val).strip().lower()
    return VENUE_CANONICAL.get(v, str(val).strip().title())

df['venue_clean'] = df['venue'].apply(normalize_venue)

# =========================================================================
# STEP 8: Title Case with acronym preservation
# =========================================================================
ACRONYMS = {'xml', 'sql', 'html', 'http', 'url', 'api', 'oltp', 'olap', 'odmg',
            'corba', 'jdbc', 'odbc', 'xjml', 'dbis', 'dom', 'sax', 'xpath',
            'xquery', 'xsl', 'xslt', 'dtd', 'rdf', 'owl', 'sparql', 'uml',
            'mpeg', 'jpeg', 'gif', 'pdf', 'csv', 'json', 'tcp', 'ip', 'udp',
            'dna', 'rna', 'gis', 'gps', 'iot', 'p2p', 'b2b', 'b2c', 'www',
            'vldb', 'sigmod', 'ieee', 'acm', 'asilomar', 'ariadne', 'tsimmis',
            'mix', 'adept', 'serf', 'tam', 'aqua', 'domino', 'ccube', 'wasa2',
            'oo', 'er', 'db', 'dbms', 'rdbms', 'nosql', 'etl', 'bi', 'dwh',
            'ii', 'iii', 'iv', 'vi', 'vii', 'viii', 'ix', 'xi', 'xii'}

def title_case_preserve(text):
    if pd.isna(text):
        return text
    words = str(text).split()
    result = []
    for i, w in enumerate(words):
        if w.lower() in ACRONYMS:
            result.append(w.upper())
        elif i == 0:
            result.append(w.capitalize())
        elif w.lower() in ('a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at',
                           'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is',
                           'was', 'are', 'were', 'vs', 'via', 'into'):
            result.append(w.lower())
        else:
            result.append(w.capitalize())
    return ' '.join(result)

df['title_final'] = df['title_clean'].apply(title_case_preserve)

# =========================================================================
# STEP 9: Build output DataFrame
# =========================================================================
out = pd.DataFrame({
    'record_id': df['id'],
    'title': df['title_final'],
    'authors': df['authors_clean'],
    'venue': df['venue_clean'],
    'year': df['year'],
    'original_title': df['title'],
    'original_authors': df['authors'],
    'original_venue': df['venue'],
})

# =========================================================================
# STEP 10: Quality scoring
# =========================================================================
def completeness_score(row):
    required = ['title', 'authors', 'venue', 'year']
    filled = sum(1 for c in required if pd.notna(row[c]) and str(row[c]).strip())
    return filled / len(required)

out['completeness_score'] = out.apply(completeness_score, axis=1)
out['publish_ready'] = out['completeness_score'] >= 0.75

# Save
out_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'tableB_normalized.csv')
out.to_csv(out_path, index=False, encoding='utf-8-sig')

# =========================================================================
# REPORT
# =========================================================================
print()
print('=== NORMALIZATION RESULTS ===')
print()

orig_df = pd.read_csv(r'C:\Users\Ravi\Downloads\tableB (3).csv', encoding='utf-8')
print('FIELD COMPLETENESS (Before -> After):')
for field in ['authors', 'venue', 'year']:
    before_null = orig_df[field].isnull().sum()
    after_null = out[field].isnull().sum()
    before_pct = before_null / len(orig_df) * 100
    after_pct = after_null / len(out) * 100
    recovered = before_null - after_null
    print(f'  {field:10s}: {before_null:5d} nulls ({before_pct:5.1f}%) -> {after_null:5d} nulls ({after_pct:5.1f}%)  [recovered: {recovered}]')
print()

print('COMPLETENESS SCORE DISTRIBUTION:')
for threshold, label in [(1.0, '100% complete'), (0.75, '75%+ complete'), (0.5, '50%+ complete'), (0.25, '25%+ complete')]:
    count = (out['completeness_score'] >= threshold).sum()
    print(f'  {label:20s}: {count:5d} records ({count/len(out)*100:.1f}%)')
print()

print(f'PUBLISH READY: {out["publish_ready"].sum()} / {len(out)} ({out["publish_ready"].sum()/len(out)*100:.1f}%)')
print()

remaining_html = out['title'].str.contains(r'&#\d+;', regex=True, na=False).sum()
remaining_html += out['authors'].str.contains(r'&#\d+;', regex=True, na=False).sum()
print(f'Remaining HTML entities: {remaining_html}')

print(f'Title length - before: median={orig_df["title"].str.len().median():.0f}, max={orig_df["title"].str.len().max()}')
print(f'Title length - after:  median={out["title"].str.len().median():.0f}, max={out["title"].str.len().max()}')
print()

# Sample before/after
print('=== SAMPLE BEFORE/AFTER (5 records) ===')
samples = [0, 4, 16, 113, 151]
for sid in samples:
    row_orig = orig_df[orig_df['id'] == sid].iloc[0]
    row_norm = out[out['record_id'] == sid].iloc[0]
    print(f'\nID {sid}:')
    print(f'  BEFORE title:   {str(row_orig["title"])[:120]}')
    print(f'  AFTER  title:   {str(row_norm["title"])[:120]}')
    print(f'  BEFORE authors: {row_orig["authors"]}')
    print(f'  AFTER  authors: {row_norm["authors"]}')
    print(f'  BEFORE venue:   {row_orig["venue"]}')
    print(f'  AFTER  venue:   {row_norm["venue"]}')
    print(f'  BEFORE year:    {row_orig["year"]}')
    print(f'  AFTER  year:    {row_norm["year"]}')

print(f'\nOutput saved: {out_path}')
