import base64
import io
import PIL


def img_to_base64(img):
    fp = io.BytesIO()
    img.save(fp, PIL.Image.registered_extensions()['.png'])
    return 'data:image/png;base64,%s' % base64.b64encode(fp.getvalue()).decode('ascii')
