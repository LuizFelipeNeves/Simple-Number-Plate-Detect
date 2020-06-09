import cv2
import os
import numpy as np
from PIL import Image

from django import forms
from django.core.files import File

from .models import Photo


PLATE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_russian_plate_number.xml".format(
	base_path=os.path.abspath(os.path.dirname(__file__)))

minArea = 200

class PhotoForm(forms.ModelForm):
    class Meta:
        model = Photo
        fields = ('file',)
        widgets = {
            'file': forms.FileInput(attrs={
                'accept': 'image/*'  # this is not an actual validation! don't rely on that!
            })
        }

    def save(self):
        photo = super(PhotoForm, self).save()

        pil_image = Image.open(photo.file)
        img = opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        height, width = img.shape[:2]
        # Scale image
        img = cv2.resize(img, (800, int((height * 800) / width)))
        # Converting to Gray Scale
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Noise removal with iterative bilateral filter(removes noise while preserving edges)
        imgGray = cv2.bilateralFilter(imgGray, 11, 17, 17)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 0)
        imgCanny = cv2.Canny(imgBlur, 150, 200)

        nPlateCascade = cv2.CascadeClassifier(PLATE_DETECTOR_PATH)
        numberP = nPlateCascade.detectMultiScale(imgGray, 1.1, 10)
        for (x, y, w, h) in numberP:
            area = w * h
            if area > minArea:
                # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
                imgCrop = imgGray[y:y + h, x:x + w]
                imgRoi = cv2.Canny(imgCrop, 150, 200)

        cv2.imwrite(photo.file.path, imgRoi)


        return photo
        

