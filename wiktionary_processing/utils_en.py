import re
from tqdm import tqdm
import xml.etree.ElementTree as ET
import mwparserfromhell

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
    if text.startswith("* ") and len(text) > 2:
        wikicode = mwparserfromhell.parse(text)
        if wikicode.filter_templates():
            items.extend([
                x.params[-1].strip() for x in wikicode.filter_templates() if x.name == "l"
            ])
        else:
            items.append(wikicode.strip_code().strip())
    return items

def parse_meaning(text):
    items = []
    if text.startswith("#") and " " in text and len(text) > 2:
        _, text = text.split(" ", 1)
        wikicode = mwparserfromhell.parse(text)
        plaintext = wikicode.strip_code().strip()
        if len(plaintext) > 2:
            items.append(plaintext)
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
    h2 = ""
    texts = []
    for line in text.split("\n"):
        if line.startswith("==") and (not line.startswith("===")) and line.endswith("=="):
            h2 = line.lower().replace(" ", "")
        if h2 == '==english==':
            texts.append(line)
    text = "\n".join(texts)
    for par in text.split("\n\n="):
        for h, f in [('===Hypernyms====', 'hypernym'), ('===Synonyms====', 'synonym')]:
            if h in par:
                res[f] += [w for line in par.split("\n") for w in parse_item(line)]
        for h, f in [('==Noun===', 'meaning'), ('==Proper noun===', 'meaning')]:
            if h in par:
                res[f] += [w.lower() for line in par.split("\n") for w in parse_meaning(line)]
    return res

def load_wiktionary(wiktionary_dump_path, vectors):
    title2docs = {key.replace(' ', '_'): val for key, val in get_title2docs(wiktionary_dump_path).items() if key in vectors}
    for title in title2docs:
        docs_info = []
        for doc in title2docs[title]:
            docs_info.append(parse_wiktionary(doc['text']))
        title2docs[title] = docs_info
    return title2docs