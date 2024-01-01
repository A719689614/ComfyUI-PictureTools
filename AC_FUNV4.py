import os
from PIL import Image,ImageEnhance,ImageOps,ImageChops,ImageDraw
import tensorflow as tf
import numpy as np
import torch

# ==============================================
# 功能函数

def ac_tensor2pilac_(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def ac_pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def read_images_from_directory(directory):
    image_list = []
    extensions = ['.jpg', '.jpeg', '.png', '.gif']  # 可以根据需要添加其他图片格式的扩展名

    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in extensions):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            image_list.append(image)

    return image_list
# ==============================================
# TODO:
class AC_FUN_ImagePath_Dont_use:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            "Image_path": ("STRING", {
                    "multiline": False,
                    }),
            "tips": ("STRING", {
                    "multiline": False, 
                    "default": '粘贴你的png格式的图片地址'}),
        }}
    # 返回结果类型
    RETURN_TYPES = ('IMAGE',)
    
    # 返回节点命名
    RETURN_NAMES = ('list_item',)
    FUNCTION = "image_path" 
    CATEGORY = "AC_FUNV4.0" 
    image_list = []

    def image_path(self,Image_path,tips=None):
            result = read_images_from_directory(Image_path)
            for x in result:
                tensors = ac_tensor2pilac_(x)
                Image = ac_pil2tensor(tensors)
                result = self.image_list.append(Image)
            return (result,)
  
# ============================================
class PictureChannels:
  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
         "Source_Image":("IMAGE",),
      "tips":("STRING",{"mutipl":False,"default":"返回图片的通道数"})
      },
    }
  RETURN_TYPES = ("INT","INT","INT","STRING")
  RETURN_NAMES = ("height","width","channels","helper")
  FUNCTION = "picturechannels"
  CATEGORY = "AC_FUNV4.0"

  def picturechannels(self,Source_Image,tips=None):
    image_shape = tf.shape(Source_Image)
    height = image_shape[0]
    width = image_shape[1]
    channels = image_shape[2]
    result = "图像尺寸：{} x {} x {}".format(height, width, channels)
    return (height,width,channels,result)
# ============================================
class PictureMix:
  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
         "image_1":("IMAGE",),
         "image_2":("IMAGE",),
         "Mix_percentage": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
      "tips":("STRING",{"mutipl":False,"default":"图像混合"})
      },
    }
  RETURN_TYPES = ("IMAGE",)
  FUNCTION = "picturemix"
  CATEGORY = "AC_FUNV4.0"

  def picturemix(self,image_1,image_2,Mix_percentage,tips=None):
    # 转换图片为PIL格式
        img_a = ac_tensor2pilac_(image_1)
        img_b = ac_tensor2pilac_(image_2)

        # Blend image
        blend_mask = Image.new(mode="L", size=img_a.size,
                               color=(round(Mix_percentage * 255)))
        blend_mask = ImageOps.invert(blend_mask)
        img_result = Image.composite(img_a, img_b, blend_mask)

        del img_a, img_b, blend_mask
    
        return (ac_pil2tensor(img_result), )

# ==============================================
# 图像去色
class AC_Image_Remove_Color:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_red": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "target_green": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "target_blue": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "replace_red": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "replace_green": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "replace_blue": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "remove_threshold": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
                "tips":("STRING",{"mutipl":False,"default":"通过阈值对图片进行去色"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_remove_color"

    CATEGORY = "AC_FUNV4.0"

    def image_remove_color(self,image,target_red=255, remove_threshold=10, target_green=255, 
                           target_blue=255, replace_red=255, replace_green=255, replace_blue=255,tips=None,):
        return (ac_pil2tensor(self.apply_remove_color(tensor2pil(image), remove_threshold, (target_red, target_green, target_blue), (replace_red, replace_green, replace_blue))),)

    def apply_remove_color(self, image, threshold=10, color=(255, 255, 255), rep_color=(0, 0, 0)):
        # Create a color image with the same size as the input image
        color_image = Image.new('RGB', image.size, color)

        # Calculate the difference between the input image and the color image
        diff_image = ImageChops.difference(image, color_image)

        # Convert the difference image to grayscale
        gray_image = diff_image.convert('L')

        # Apply a threshold to the grayscale difference image
        mask_image = gray_image.point(lambda x: 255 if x > threshold else 0)

        # Invert the mask image
        mask_image = ImageOps.invert(mask_image)

        # Apply the mask to the original image
        result_image = Image.composite(
            Image.new('RGB', image.size, rep_color), image, mask_image)

        return result_image
    
# ==============================================
class AC_Image_Batch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_1": ("IMAGE",),
                "images_2": ("IMAGE",),
          
                # "images_e": ("IMAGE",),
                # "images_f": ("IMAGE",),
                # Theoretically, an infinite number of image input parameters can be added.
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_batch"
    CATEGORY = "AC_FUNV4.0"

    def image_batch(self,images_1=None,images_2=None):
        result = images_1+images_2
        return (result,)
# ==============================================
# TODO
class AC_ImageContrast:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "contrast": ("FLOAT",
                               {"min": -50.00, "max": 50.00, "step": 1.00}
                               ),
                "tips":("STRING",{"mutilin":False,"default":"调整图片的对比度"})
            },
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_Contrast"
    CATEGORY = "AC_FUNV4.0"

    def image_Contrast(self,image=None,contrast=None,tips=None):
        # 转换为 PIL 图像对象
        image_pil = ac_tensor2pilac_(image)
        # 调整对比度
        enhanced_image_pil = ImageEnhance.Contrast(image_pil).enhance(contrast)
        # 转换回 Tensor 对象
        enhanced_image_tensor = ac_pil2tensor(enhanced_image_pil)

        return (enhanced_image_tensor,)
# ==============================================
class AC_ImageBrightness:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT",
                               {"min": -50.00, "max": 50.00, "step": 1.00}
                               ),
                "tips":("STRING",{"mutilin":False,"default":"调整图片的亮度/暗度"})
            },
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_brightness"
    CATEGORY = "AC_FUNV4.0"

    def image_brightness(self,image=None,brightness=None,tips=None):
        # 转换为 PIL 图像对象
        image_pil = ac_tensor2pilac_(image)
        # 调整亮度
        enhanced_image_pil = ImageEnhance.Brightness(image_pil).enhance(brightness)
        # 转换回 Tensor 对象
        enhanced_image_tensor = ac_pil2tensor(enhanced_image_pil)

        return (enhanced_image_tensor,)
# ==============================================
class AC_ImageDrow:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "alpha": ("FLOAT",
                               {"min": 0.00, "max": 1.00, "step": 0.01}
                               ),
                "tips":("STRING",{"mutilin":False,"default":"调整图片叠加模式"})
            },
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_drow"
    CATEGORY = "AC_FUNV4.0"

    def image_drow(self,image_1,image_2,alpha,tips=None):
        # 转换为 PIL 图像对象
        P1 = ac_tensor2pilac_(image_1)
        P2 = ac_tensor2pilac_(image_2)

        # 混合模式
        blended_image = Image.blend(P1, P2, alpha)
        
        # 转换回 Tensor 对象
        enhanced_image_tensor = ac_pil2tensor(blended_image)

        return (enhanced_image_tensor,)
# ==============================================
class AC_ImageBalance:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "factor": ("FLOAT",
                               {"min": 0.00, "max": 1.00, "step": 0.01}
                               ),
                "tips":("STRING",{"mutilin":False,"default":"色彩平衡(饱和度)"})
            },
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_balance"
    CATEGORY = "AC_FUNV4.0"

    def image_balance(self,image_1,factor,tips=None):
        # 转换为 PIL 图像对象
        P1 = ac_tensor2pilac_(image_1)

        # 创建 ImageEnhance.Color 对象并进行颜色平衡调整
        enhancer = ImageEnhance.Color(P1)
        balanced_image = enhancer.enhance(factor)

        
        # 转换回 Tensor 对象
        enhanced_image_tensor = ac_pil2tensor(balanced_image)

        return (enhanced_image_tensor,)
# ==============================================
class AC_ImageCopy:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "tips":("STRING",{"mutilin":False,"default":"复制图像"})
            },
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_copy"
    CATEGORY = "AC_FUNV4.0"

    def image_copy(self,image_1,tips=None):
        # 转换为 PIL 图像对象
        P1 = ac_tensor2pilac_(image_1)
        # 对图片进行复制
        pil_image1 = P1.copy()
        # 转换回 Tensor 对象
        enhanced_image_tensor_1 = ac_pil2tensor(pil_image1)
        return (enhanced_image_tensor_1,)
# ==============================================
class AC_Image_invert:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "tips":("STRING",{"mutilin":False,"default":"图片反向"})
            },
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_invert"
    CATEGORY = "AC_FUNV4.0"

    def image_invert(self,image_1,tips=None):
        # 转换为 PIL 图像对象
        P1 = ac_tensor2pilac_(image_1)
        # 对图片进行复制
        pil_image1 = ImageOps.invert(P1)
        # 转换回 Tensor 对象
        enhanced_image_tensor_1 = ac_pil2tensor(pil_image1)
        return (enhanced_image_tensor_1,)
# ==============================================
class AC_ImageCrop:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT",{"min": 0, "max": 9999, "step": 1}),
                "right": ("INT",{"min": 0, "max": 9999, "step": 1,"default":256}),
                "top": ("INT",{"min": 0, "max": 9999, "step": 1}),
                "bottom": ("INT",{"min": 0, "max": 9999, "step": 1,"default":256}),
                      }}
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NAMES = ("image",)
    FUNCTION = "image_crop"
    CATEGORY = "AC_FUNV4.0"
    def image_crop(self,image,left=0,right=256,top=0,bottom=256):
        image = tensor2pil(image)
        img_width, img_height = image.size
        crop_top = max(top, 0)
        crop_left = max(left, 0)
        crop_bottom = min(bottom, img_height)
        crop_right = min(right, img_width)
        crop_width = crop_right - crop_left
        crop_height = crop_bottom - crop_top
        if crop_width <= 0 or crop_height <= 0:
            raise ValueError("Error: crop_width and crop_height")
        
        # Crop the image and resize
        crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
        crop = crop.resize((((crop.size[0] // 8) * 8), ((crop.size[1] // 8) * 8)))
        
        return (ac_pil2tensor(crop),)
# ================================================================
class AC_ImageSize:
    RESOLUTIONS = ["Cinema (1536x640)", "Widescreen (1344x768)", "Photo (1216x832)", "TV (1152x896)", "Square (1024x1024)"]
    ASPECT = ["Landscape", "Portrait"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "resolution": (AC_ImageSize.RESOLUTIONS, {"default": "Square (1024x1024)"}),
                    "aspect": (AC_ImageSize.ASPECT, {"default": "Portrait"})
                    },
                }

    RETURN_TYPES = ("INT", "INT", )
    RETURN_NAMES = ("width", "height", )
    FUNCTION = "get_value"

    CATEGORY = "AC_FUNV4.0"

    def get_value(self, resolution, aspect, ):
        if resolution == "Cinema (1536x640)":
            if aspect == "Landscape":
                return (1536, 640)
            else:
                return (640, 1536)
        if resolution == "Widescreen (1344x768)":
            if aspect == "Landscape":
                return (1344, 768)
            else:
                return (768, 1344)
        if resolution == "Photo (1216x832)":
            if aspect == "Landscape":
                return (1216, 832)
            else:
                return (832, 1216)
        if resolution == "TV (1152x896)":
            if aspect == "Landscape":
                return (1152, 896)
            else:
                return (896, 1152)
        if resolution == "Square (1024x1024)":
            return (1024, 1024)
# ===================================================================================
Image_mode = ["red", "green", "blue"]
class Image_channel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            "image":("IMAGE",),
            "mode":(Image_mode,),
            "tips":("STRING",{"mutilin":False,"default":"图像的红绿蓝通道图"})
            },
        }
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NAMES = ("image",)
    FUNCTION = "image_channel"
    CATEGORY = "AC_FUNV4.0"

    def image_channel(self,mode,image,tips=None):
        # tensor转换成pil
        image_pil = ac_tensor2pilac_(image)
        # 分离通道
        r, g, b = image_pil.split()

        # 提取红色、绿色、蓝色通道
        red_image = Image.merge("RGB", (r, Image.new("L", image_pil.size, 0), Image.new("L", image_pil.size, 0)))
        green_image = Image.merge("RGB", (Image.new("L", image_pil.size, 0), g, Image.new("L", image_pil.size, 0)))
        blue_image = Image.merge("RGB", (Image.new("L", image_pil.size, 0), Image.new("L", image_pil.size, 0), b))
        # 判定条件
        if mode == "red":
            return (ac_pil2tensor(red_image),)
        if mode == "green":
            return (ac_pil2tensor(green_image),)
        if mode == "blue":
            return (ac_pil2tensor(blue_image))
#do some processing on the image, in this example I just invert it   
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {

    
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {

}

if __name__=="__main__":
   pass

   



