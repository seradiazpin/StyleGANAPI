import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore, storage
from config import Settings
settings = Settings()

class FireBase(object):
    def __init__(self):
        self.credentials = credentials.Certificate(settings.firebase_config)
        self.bucket = settings.firebase_bucket
        if not firebase_admin._apps:
            firebase_admin.initialize_app(self.credentials)
        self.db = firestore.client()
        self.st = storage.bucket(self.bucket)

    def create_file(self, collection, data):
        ref = self.db.collection(collection).add(data)
        self.store_file(data["link"], data["path"])
        self.store_file(data["link_small"], data["path_small"])

    def create(self, collection, data):
        ref = self.db.collection(collection).add(data)

    def read(self, collection, element=None, size=16):
        collection_ref = self.db.collection(collection).order_by(u'time', direction=firestore.Query.DESCENDING).limit(
            size)
        if element is not None:
            doc = self.db.collection(collection).document(element).get()
            collection_ref = collection_ref.start_after(doc)
        return collection_ref.stream()

    def read_query(self, collection, attribute, condition, value):
        return self.db.collection(collection).where(attribute, condition, value).get()

    def update(self, collection):
        ref = self.db.reference(collection)
        ref.set({'test': 'test'})

    def delete(self, collection):
        ref = self.db.reference(collection)
        ref.set({'test': 'test'})

    def get_file(self, file):
        blob = self.st.blob(file)
        blob.download_to_filename("static/firebase.png")

    def get_file_url(self, file):
        blob = self.st.blob(file)
        return blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')

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
