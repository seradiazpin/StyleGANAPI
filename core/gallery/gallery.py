from data.firebase.firebase import FireBase


def load_gallery(document, size):
    fb = FireBase()
    collection_ref = fb.read(u'Generated', document, size)
    d = {"img": []}
    for doc in collection_ref:
        print(u'{} => {}'.format(doc.id, doc.to_dict()))
        url = doc.to_dict()
        url["link_small"] = fb.get_file_url(file=url["link_small"])
        url["link"] = fb.get_file_url(file=url["link"])
        url["id"] = doc.id
        d["img"].append(url)
    return d
