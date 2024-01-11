from .AC_FUNV4 import *
from .AC_FUN_PATH import *
from .extend_nodes import *
from .AC_Optical_Flow import *
from .AC_RMB import*

NODE_CLASS_MAPPINGS = {
    # 啊程V4.0
    "📱批量图片LT(问题测试)":AC_FUN_ImagePath_Dont_use,
    "✅批量图像BA":AC_Batch,
    "💹批量保存SV":AC_Save,
    "🎦图像通道图":PictureChannels,
    "🆎图像混合Mix":PictureMix,
    "➿图像去色RC":AC_Image_Remove_Color,
    "🔤图像合并MG":AC_Image_Batch,
    "🔀图像亮度BT":AC_ImageBrightness,
    "🆙图像对比度IC":AC_ImageContrast,
    "⏬图像反向":AC_Image_invert,
    "🆖图像裁切CP":AC_ImageCrop,
    "🔄图像叠加模式ID":AC_ImageDrow,
    "🔛图像色彩平衡":AC_ImageBalance,
    "🛐图像复制CP":AC_ImageCopy,
    "♌Web设备模板":AC_ImageSize,
    "🔚图像红绿蓝通道":Image_channel,
    "📄加载图片64位编码": LoadImageBase64,
    "📃加载遮罩64位编码": LoadMaskBase64,
    "🌐发送web图片接口": SendImageWebSocket,
    "❌简易裁切图片": CropImage,
    "♎应用蒙版到图像": ApplyMaskToImage,
    "💻计算机图像选项": ComputeOpticalFlow,
    "📹应用图形选项": ApplyOpticalFlow,
    "🎴预览计算机图形选项": VisualizeOpticalFlow,
    "🔃图片内容溶解":AC_Remove_Background,
    "🆖图片移除背景":AC_Remove_Rembg
    



}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {

}

__all__ = ["NODE_CLASS_MAPPINGS"]



print("Cc啊程、AC_FUNV4.0图像处理模组加载......")