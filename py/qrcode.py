from .imagefunc import *

class CreateQRCode:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "size": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "border": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "text": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = 'create_qrcode'
    CATEGORY = '😺dzNodes/LayerUtility/SystemIO'

    def create_qrcode(self, size, border, text):
        import qrcode
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=20,
            border=border,
        )
        qr.add_data(text.encode('utf-8'))
        qr.make(fit=True)
        ret_image = qr.make_image(fill_color="black", back_color="white")
        ret_image = ret_image.resize((size, size), Image.BICUBIC)

        return (pil2tensor(ret_image.convert("RGB")), )

class DecodeQRCode:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE",),
                "pre_blur": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("string", )
    FUNCTION = 'decode_qrcode'
    CATEGORY = '😺dzNodes/LayerUtility/SystemIO'

    def decode_qrcode(self, image, pre_blur):
        ret_texts = []
        from pyzbar.pyzbar import decode
        for i in image:
            _image = torch.unsqueeze(i, 0)
            _image = tensor2pil(_image)
            if pre_blur:
                _image = gaussian_blur(_image, pre_blur)
            qrmessage = decode(_image)
            if len(qrmessage) > 0:
                ret_texts.append(qrmessage[0][0].decode('utf-8'))
            else:
                ret_texts.append("Cannot recognize QR")

        return (ret_texts, )

NODE_CLASS_MAPPINGS = {
    "LayerUtility: CreateQRCode": CreateQRCode,
    "LayerUtility: DecodeQRCode": DecodeQRCode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: CreateQRCode": "LayerUtility: Create QRCode",
    "LayerUtility: DecodeQRCode": "LayerUtility: Decode QRCode"
}