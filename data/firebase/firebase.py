import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore, storage


class FireBase(object):
    def __init__(self):
        self.credentials = credentials.Certificate("./data/firebase/firebasekey.json")
        self.bucket = "thumbnailgenerator-c1e1b.appspot.com"
        if not firebase_admin._apps:
            firebase_admin.initialize_app(self.credentials)
        self.db = firestore.client()
        self.st = storage.bucket(self.bucket)

    def create(self, collection, data):
        ref = self.db.collection(collection).add(data)
        self.store_file(data["link"], data["path"])

    def read(self, collection):
        collection_ref = self.db.collection(collection)
        return collection_ref.stream()

    def update(self, collection):
        ref = self.db.reference(collection)
        ref.set({'test': 'test'})

    def delete(self, collection):
        ref = self.db.reference(collection)
        ref.set({'test': 'test'})

    def get_file(self, file):
        blob = self.st.blob(file)
        blob.download_to_filename("static/firebase.png")

    def store_file(self, file, file_path):
        blob = self.st.blob(file)
        blob.upload_from_filename(file_path)

    def print_file(self, file):
        blob = self.st.blob(file)
        print(blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET'))

    def print_collection(self, collection):
        firebase_admin.initialize_app(self.credentials)
        db = firestore.client()

        collection_ref = self.db.collection(collection)
        for doc in collection_ref:
            print(u'{} => {}'.format(doc.id, doc.to_dict()))


if __name__ == '__main__':
    FireBase().get_file("generated/c897c782-289f-441c-937f-8ea42c2214ec.png")