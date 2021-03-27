import pytesseract

class OCR:
    def __init__(self):
        self.custom_config = r'--oem 3 --psm 6'

    def readimage(self,address):
        s = pytesseract.image_to_string(address, config=self.custom_config)
        return s


# Adding custom options

if __name__ == "__main__":
    cc = OCR()
    print(cc.readimage("cv_example.png"))

