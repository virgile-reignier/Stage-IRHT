import spacy
from spacy.kb import KnowledgeBase

import csv
from pathlib import Path


def load_entities():
    """ Helper function to read in the pre-defined entities we want to disambiguate to. """
    input_dir = Path.cwd().parent / "input"
    entities_loc = input_dir / "kb_Vienne.csv"

    names = dict()
    descriptions = dict()
    with entities_loc.open("r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter="\t")
        for row in csvreader:
            qid = row[0]
            name = row[1]
            desc = row[15]
            names[qid] = name
            descriptions[qid] = desc
    return names, descriptions


if __name__ == "__main__":
    nlp = spacy.load("/home/reignier/Bureau/Entity-linking/spacy-ner-irht-teklia/multi-home-c3po4-LOC-model-best")
    text = "Actum Vienne decima, die april, anno Domini m. ccc. duodecimo. Per dominum Marregi"
    doc = nlp(text)
    name_dict, desc_dict = load_entities()

    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=0)

    for qid, desc in desc_dict.items():
        desc_doc = nlp(desc)
        desc_enc = desc_doc.vector
        print(desc_enc)
        kb.add_entity(entity=qid, entity_vector=desc_enc, freq=342)

    for qid, name in name_dict.items():
        kb.add_alias(alias=name, entities=[qid], probabilities=[1])

    qids = name_dict.keys()
    probs = [0.3 for qid in qids]
    kb.add_alias(alias="Vienne", entities=qids, probabilities=probs)

    print(f"Entities in the KB: {kb.get_entity_strings()}")
    print(f"ALiases in the KB: {kb.get_alias_strings()}")
    print()
    print(f"Candidates for 'Roy Stanley Emerson': {[c.entity_ for c in kb.get_candidates('Roy Stanley Emerson')]}")
    print(f"Candidates for 'Emerson': {[c.entity_ for c in kb.get_candidates('Emerson')]}")
    print(f"Candidates for 'Sofie': {[c.entity_ for c in kb.get_candidates('Sofie')]}")
