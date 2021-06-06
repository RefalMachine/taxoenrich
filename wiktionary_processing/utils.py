import re
from tqdm import tqdm
import xml.etree.ElementTree as ET

def get_title2docs(dump_path, title_list=None):
    title2doc = {}
    doc = {}
    fields = {
        "timestamp": "timestamp",
        "title": "title",
        "text": "text",
        "redirect title": "redirect_title",
    }
    cnt = 0
    for _, elem in tqdm(ET.iterparse(dump_path, events=("end",))):
        prefix, has_namespace, postfix = elem.tag.partition('}')
        tag = postfix if postfix else prefix
        if tag in fields:
            doc[fields[tag]] = elem.text
        if tag == "page":
            elem.clear()
            cnt += 1
            if title_list is None or doc['title'].lower() in title_list:
                title2doc[doc["title"]] = doc
            doc = {}

    ltitle2docs = {}
    for x in title2doc.keys():
        ltitle2docs.setdefault(x.lower(), []).append(title2doc[x])

    return ltitle2docs

def clean_markup(text):
    return text.replace("[[", "").replace("]]", "").replace("{{aslinks|", "")

def parse_item(text):
    items = []
    if text.startswith("# ") and len(text) > 2:
        items.extend([
            clean_markup(x).replace("?", "").replace(";", "").replace("'", "").strip() 
            for x in re.split(',|;', text[2:]) if x not in {'-', '?', '—', ''}
        ])
    return items

def parse_translation(trans):
    res = {}
    for line in trans.split('\n'):
        if line.startswith('|'):
            l, r = line.split('=')
            res[l[1:]] = r.replace('[[', '').replace(']]', '')
    return res

def parse_wiktionary(text):
    res = {'hypernym': [], 'synonym': [], 'meaning': []}
    h1 = ""
    texts = []
    for line in text.split("\n"):
        if line.startswith("= ") and line.endswith(" ="):
            h1 = line
        if h1 == '= {{-ru-}} =':
            texts.append(line)
    text = "\n".join(texts)
    for par in text.split("\n\n"):
        for h, f in [('==== Гиперонимы ====', 'hypernym'), ('==== Синонимы ====', 'synonym')]:
            if h in par:
                res[f] += [w.replace(' ', '_').lower() for line in par.split("\n") for w in parse_item(line)]
        for h, f in [('==== Значение ====', 'meaning')]:
            if h in par:
                for line in par.split('\n'):
                    if line.startswith('# ') and len(line) > 2:
                        res[f] += [clean_markup(line[2:]).lower()]
                #res[f] += [clean_markup(line[2:]).split() for line in par.split("\n") if line.startswith('# ') and len(line) > 2]
        #res[f] = [item for sublist in res[f] for item in sublist]
        #print(res[f])
        #if '=== Перевод ===' in par:
        #    res['translation'] = par.replace('=== Перевод ===\n', '')
    return res


def load_wiktionary(wiktionary_dump_path, vectors):
    title2docs = {key.replace(' ', '_'): val for key, val in get_title2docs(wiktionary_dump_path).items() if key in vectors}
    for title in title2docs:
        docs_info = []
        for doc in title2docs[title]:
            docs_info.append(parse_wiktionary(doc['text']))
        title2docs[title] = docs_info
    return title2docs
