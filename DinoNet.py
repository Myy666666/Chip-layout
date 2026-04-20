
import cv2
import numpy as np
import dino
import torch
import matplotlib.pyplot as plt 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
# 全局随机种子设置，保证特征一致性
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
def extract_feature_vector(img_path, dino_net, patch_h, patch_w, feat_dim,num, device='cuda'):
    import torchvision.transforms as T
    from PIL import Image
    # 预处理
    transform = T.Compose([
        # 去除随机高斯模糊，保证输入一致性
        # T.GaussianBlur(9, sigma=(0.1, 2.0)),
        T.Resize((patch_h * 14, patch_w * 14)),
        T.CenterCrop((patch_h * 14, patch_w * 14)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img = Image.open(img_path).convert('RGB')
    imgs_tensor = torch.zeros(1, 3, patch_h * 14, patch_w * 14).to(device)
    imgs_tensor[0] = transform(img)[:3]
    with torch.no_grad():
        features_dict = dino_net.model.forward_features(imgs_tensor,num)
        features = features_dict['x_norm_patchtokens']
    features = features.reshape(1 * patch_h * patch_w, feat_dim).cpu().numpy()
    # PCA降维
    reduced_features = DinoNet.pca_reduce(features, out_dim=3)
    # reshape为(16, 16, 3)
    reshaped = reduced_features.reshape(16, 16, 3)
    # --- 热力图叠加原图并保存 ---
    import os
    save_dir = os.path.dirname(img_path)
    # 生成热力图（取mean通道）
    heatmap = DinoNet.feature_to_heatmap(reshaped, method='mean')
    # 读取原图并resize到热力图尺寸
    import cv2
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        from PIL import Image
        orig_img = np.array(Image.open(img_path).convert('RGB'))
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    # 将热力图resize到原图尺寸
    heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_LINEAR)
    # 叠加显示（与原图同尺寸）
    overlay = DinoNet.overlay_heatmap_on_image(orig_img, heatmap_resized, alpha=0.8)
    # 保存融合结果
    target_dir = os.path.join(os.path.dirname(__file__), "outputs", "heatmaps")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    overlay_path = os.path.join(target_dir, 'heatmap_overlay_'+str(num) + os.path.basename(img_path))
    overlay_with_colorbar_path = os.path.join(
        target_dir, 'heatmap_overlay_colorbar_' + str(num) + os.path.basename(img_path)
    )
    print('overlay dtype:', overlay.dtype, 'shape:', overlay.shape, 'min:', overlay.min(), 'max:', overlay.max())
    print(overlay_path)
    cv2.imwrite(overlay_path, overlay)
    DinoNet.save_overlay_with_colorbar(
        overlay_bgr=overlay,
        save_path=overlay_with_colorbar_path,
        title='Feature Heatmap Overlay',
        cmap='jet'
    )
    # 可选：matplotlib显示
    import matplotlib.pyplot as plt
    # plt.figure(figsize=(6, 6))
    # plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.title('Feature Heatmap Overlay')
    # plt.show()
    # 4x4池化（平均池化）
    pooled = torch.nn.functional.avg_pool2d(torch.from_numpy(reshaped.transpose(2,0,1)[None]), kernel_size=4, stride=4)
    pooled_np = pooled.squeeze(0).permute(1,2,0).numpy()
    # 平铺为一维向量
    flattened = pooled_np.flatten()
    print(flattened.shape)
    return flattened
#创建dino模型，目的是先提取特征
class DinoNet(nn.Module):
    @staticmethod
    def feature_to_heatmap(feature_map, method='mean', amplify=1.0):
        """
        将特征张量转换为单通道热力图并应用伪彩色。
        Args:
            feature_map: (H, W, C) numpy数组
            method: 'mean' 或 'max'，决定如何降为单通道
            amplify: 放大系数，增强特征对比度
        Returns:
            heatmap: (H, W, 3) 彩色热力图
        """
        import cv2
        if method == 'mean':
            heat = feature_map.mean(axis=-1)
        elif method == 'max':
            heat = feature_map.max(axis=-1)
        else:
            raise ValueError('method must be mean or max')
        # 放大特征对比度
        heat = heat * amplify
        # 归一化到0-1
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        heat = (heat * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        return heatmap

    @staticmethod
    def overlay_heatmap_on_image(image, heatmap, alpha=0.5):
        """
        将热力图叠加到原图上。
        Args:
            image: 原始图片 (H, W, 3) numpy
            heatmap: 热力图 (H, W, 3) numpy
            alpha: 叠加权重
        Returns:
            overlay: 融合后的图片
        """
        import cv2
        if image.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        overlay = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
        return overlay

    @staticmethod
    def save_overlay_with_colorbar(overlay_bgr, save_path, title='Heatmap Overlay', cmap='jet'):
        """
        保存带色带说明的热力图叠加结果。

        色带从蓝色到红色：
        - 蓝色表示低响应/低强度
        - 红色表示高响应/高强度
        """
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import cv2

        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        bg_color = (225 / 255.0, 239 / 255.0, 214 / 255.0)

        fig = plt.figure(figsize=(8, 9), facecolor=bg_color)
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[16, 1], hspace=0.12)

        ax_img = fig.add_subplot(gs[0])
        ax_img.set_facecolor(bg_color)
        ax_img.imshow(overlay_rgb)
        ax_img.set_title(title)
        ax_img.axis('off')

        ax_cbar = fig.add_subplot(gs[1])
        ax_cbar.set_facecolor(bg_color)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = fig.colorbar(sm, cax=ax_cbar, orientation='horizontal')
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['0', '1'])
        cbar.outline.set_linewidth(1.0)
        ax_cbar.tick_params(axis='x', labelsize=12, length=0, pad=2)

        plt.tight_layout(pad=0.6)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

    @staticmethod
    def pca_reduce(features: np.ndarray, out_dim: int = 3) -> np.ndarray:
        """
        对特征向量进行PCA降维。
        Args:
            features: 输入特征，shape=(N, D) 或 (D,)
            out_dim: 降维后维度
        Returns:
            降维后的特征，shape=(N, out_dim) 或 (out_dim,)
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        pca = PCA(n_components=out_dim)
        reduced = pca.fit_transform(features)
        if reduced.shape[0] == 1:
            return reduced[0]
        return reduced
    @staticmethod
    def force_resize224(img):
        """
        强制将图片resize为(224,224)，输出3通道彩色图像。
        """
        import cv2
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        return img

    def detect_keypoints(self, image, nms_radius=4, threshold=0.2, max_keypoints=-1, response_type='l2'):
        """
        基于DINO特征图的关键点检测。
        Args:
            image: 输入图片 (H, W, 3) numpy 或 (3, H, W) torch/numpy
            nms_radius: NMS半径（像素，特征图尺度）
            threshold: 响应阈值
            max_keypoints: 最多保留关键点数，-1为不限
            response_type: 'l2'（默认，特征L2范数）或'channel_max'（最大通道响应）
        Returns:
            keypoints: (N, 2) numpy数组，xy坐标（原图尺度）
            scores: (N,) numpy数组，响应分数
        """
        import torch
        import numpy as np
        import torch.nn.functional as F
        # 1. 提取特征图 (C, Hf, Wf)
        # 强制resize为(224,224)
        image = self.force_resize224(image)
        feat_map = self.extract_feature(image)
        if isinstance(feat_map, torch.Tensor):
            feat_map = feat_map.detach().cpu()
        # 2. 计算响应图
        if response_type == 'l2':
            response = torch.norm(feat_map, p=2, dim=0)  # (Hf, Wf)
        elif response_type == 'channel_max':
            response = torch.max(feat_map, dim=0)[0]
        else:
            raise ValueError('Unknown response_type')
        # 3. NMS（简单max-pooling实现）
        response = response.unsqueeze(0).unsqueeze(0)  # (1,1,Hf,Wf)
        nms_mask = (response == F.max_pool2d(response, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius))
        response = response.squeeze()
        nms_mask = nms_mask.squeeze()
        # 4. 阈值筛选
        mask = (response > threshold) & nms_mask
        yx = torch.nonzero(mask, as_tuple=False)  # (N,2) [y,x]
        scores = response[mask]
        # 5. 排序与裁剪
        if max_keypoints > 0 and scores.numel() > max_keypoints:
            topk = torch.topk(scores, max_keypoints)
            idx = topk.indices
            yx = yx[idx]
            scores = scores[idx]
        # 6. 坐标映射回原图
        # 计算下采样率
        Hf, Wf = response.shape
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] in [1,3]:
                H, W = image.shape[1:]
            else:
                H, W = image.shape[:2]
        elif isinstance(image, torch.Tensor):
            if image.ndim == 3 and image.shape[0] in [1,3]:
                H, W = image.shape[1:]
            else:
                H, W = image.shape[-2:]
        else:
            raise ValueError('Unknown image type')
        scale_y = H / Hf
        scale_x = W / Wf
        xy = torch.stack([yx[:,1]*scale_x, yx[:,0]*scale_y], dim=1)  # (N,2)
        return xy.numpy(), scores.numpy()
    def __init__(self, cpt_path=None, feature_layer=1, load_ckpt=True):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.feature_layer = feature_layer
        # 获取模型
        # 修正为ViT-L/14结构，确保与权重匹配
        # self.model = dino.vit_base()
        self.model = dino.vit_large()
        # 只有在需要加载权重且路径有效时才加载
        if load_ckpt and cpt_path:
            state_dict_raw = torch.load(cpt_path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(state_dict_raw)
        # 否则直接跳过权重加载，模型结构始终初始化
        self.model = self.model.to(self.device)
        # 开启评估模式，不让其训练
        self.model.eval()
        self.image_size_max = 630

        self.h_down_rate = self.model.patch_embed.patch_size[0]
        self.w_down_rate = self.model.patch_embed.patch_size[1]

     #修改图片尺寸，将长和宽改成14的倍数，方便后续dino分解块,图片从这里开始输入，维度[w,h,3]
    # def _resize_input_images(self, images: np.ndarray, interpolation=cv2.INTER_LINEAR):
    #     """批量处理图片尺寸，使其能被down_rate整除。

    #     Args:
    #         images: (B, H, W, C) 或 (H, W, C) 的numpy数组
    #         interpolation: 插值方法

    #     Returns:
    #         调整尺寸后的图片数组
    #     """
    #     # 如果是单张图片，增加批次维度
    #     if images.ndim == 3:
    #         images = images[np.newaxis, ...]
    #         was_single = True
    #     else:
    #         was_single = False

    #     batch_size = images.shape[0]
    #     resized_images = []

    #     for i in range(batch_size):
    #         image = images[i]
    #         h_image, w_image = image.shape[:2]
    #         h_larger_flag = h_image > w_image
    #         large_side_image = max(h_image, w_image)

    #         # 调整目标尺寸
    #         if large_side_image > self.image_size_max:
    #             if h_larger_flag:
    #                 h_image_target = self.image_size_max
    #                 w_image_target = int(self.image_size_max * w_image / h_image)
    #             else:
    #                 w_image_target = self.image_size_max
    #                 h_image_target = int(self.image_size_max * h_image / w_image)
    #         else:
    #             h_image_target = h_image
    #             w_image_target = w_image

    #         # 确保能被下采样率整除
    #         h = max(1, h_image_target // self.h_down_rate)
    #         w = max(1, w_image_target // self.w_down_rate)
    #         h_resize, w_resize = h * self.h_down_rate, w * self.w_down_rate

    #         # 调整尺寸
    #         resized = cv2.resize(image, (w_resize, h_resize), interpolation=interpolation)
    #         resized_images.append(resized)

    #     # 合并为批次
    #     resized_batch = np.stack(resized_images, axis=0)

    #     # 如果是单张输入，去除批次维度
    #     if was_single:
    #         resized_batch = resized_batch.squeeze(0)
    #     return resized_batch
    
    #将图片进行归一化，转化为tensor张量，变化一下维度 [3,w,h]
    # def _process_images(self, image: np.ndarray) -> torch.Tensor:
    #     """Turn image into pytorch tensor and normalize it."""
    #     # mean = np.array([0.485, 0.456, 0.406])
    #     # std = np.array([0.229, 0.224, 0.225])
    #     mean = np.array([0.5, 0.5, 0.5])
    #     std = np.array([0.5, 0.5, 0.5])
    #     image_processed = image / 255.0
    #     image_processed = (image_processed - mean) / std
    #     image_processed = torch.from_numpy(image_processed).permute(2, 0, 1)
    #     image_processed=2*image_processed-1
    # #     return image_processed
    # def _process_images(self, images: np.ndarray) -> torch.Tensor:
    #     """批量处理图片，转换为PyTorch张量并归一化。

    #     Args:
    #         images: (B, H, W, 3) 或 (H, W, 3) 的numpy数组

    #     Returns:
    #         处理后的张量: (B, 3, H, W) 或 (3, H, W)
    #     """
    #     # 检查输入维度
    #     if images.ndim == 3:
    #         # 单张图片: (H, W, 3)
    #         images = images[np.newaxis, ...]  # 增加批次维度 -> (1, H, W, 3)
    #         was_single = True
    #     elif images.ndim == 4:
    #         # 批量图片: (B, H, W, 3)
    #         was_single = False
    #     else:
    #         raise ValueError(f"输入维度必须是3或4维，实际是 {images.ndim} 维,，问题出现在DinoNet文件里面")

    #     # 归一化参数
    #     mean = np.array([0.5, 0.5, 0.5])
    #     std = np.array([0.5, 0.5, 0.5])

    #     # 批量归一化
    #     images = images.astype(np.float32) / 255.0  # [0, 255] -> [0, 1]
    #     images = (images - mean) / std  # 标准化

    #     # 转换为张量并调整维度: (B, H, W, 3) -> (B, 3, H, W)
    #     images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)

    #     # 额外的归一化: [0, 1] -> [-1, 1]
    #     images_tensor = 2 * images_tensor - 1
    #     # 如果是单张输入，去除批次维度
    #     if was_single:
    #         images_tensor = images_tensor.squeeze(0)  # (1, 3, H, W) -> (3, H, W)
    #     return images_tensor
    
    def __call__(self, image) -> np.ndarray:
        return self.forward(image)
# sdrgs:
    def extract_feature(self, image):
        """Extracts features from image.

        Args:
            image: (B, 3, H, W) 或 (3, H, W) torch tensor或numpy，已归一化
        Returns:
            features: (B, C, H//14, W//14) 或 (C, H//14, W//14) torch tensor image features.
        """
        device = next(self.model.parameters()).device
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.ndim == 3 and image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            # (H, W, 3) -> (3, H, W)
            image = torch.from_numpy(image).float().permute(2, 0, 1)
        image = image.to(device)
        # 自动补 batch 维度
        was_single = False
        if image.ndim == 3:
            image = image.unsqueeze(0)
            was_single = True
        b, _, h_origin, w_origin = image.shape

        #获取第几层的特征，后面为[0],表示返回数组的第一层特征
        out = self.model.get_intermediate_layers(image, n=self.feature_layer)
        out = out[0]
        h = int(h_origin / self.h_down_rate)
        w = int(w_origin / self.w_down_rate)

        dim = out.shape[-1]
        #每一层出来的特征向量维度是(b,l,c)要转化为:(b,w,h,c)
        out_temp = out.reshape(b, h, w, dim).permute(0, 3, 1, 2)
        out_temp = 2 * out_temp - 1
        if was_single:
            out_temp = out_temp.squeeze(0)
        return out_temp
#     def extract_feature(self, image):
#         """Extracts features from image.

# sdrgs:
#           image: (B, 3, H, W) torch tensor, normalized with ImageNet mean/std.

#         Returns:
#           features: (B, C, H//14, W//14) torch tensor image features.
#         """
#         b, _, h_origin, w_origin = image.shape
        
#         #创建特征字典
#         feature_dist={}
        
#         #获取第几层的特征，后面为[0],表示返回数组的第一层特征
#         out = self.model.get_intermediate_layers(image, n=self.feature_layer)
#         h = int(h_origin / self.h_down_rate)
#         w = int(w_origin / self.w_down_rate)
#         for i in range(self.feature_layer):
#             k=12-self.feature_layer+i
#             out_temp=out[i]
#             dim = out_temp.shape[-1]
#             #每一层出来的特征向量维度是(b,l,c)要转化为:(b,w,h,c)
#             out_temp = out_temp.reshape(b, h, w, dim).permute(0, 3, 1, 2).detach()
#             # out_temp=out_temp.squeeze(0).permute(1, 2, 0).cpu().numpy()

#             out_temp=2*out_temp-1
#             feature_dist[k]=out_temp
            
#         #返回特征字典值
#         return feature_dist
    
    
    def forward(self, image) -> np.ndarray:
        """Feeds image through DINO ViT model to extract features.

        Args:
          image: (H, W, 3) numpy array, decoded image bytes, value range [0, 255].

        Returns:
          features: (H // 14, W // 14, C) numpy array image features.
        """

        features_dist=self.extract_feature(image)
        print(features_dist)
        return features_dist
def auto_resize_to_patch_multiple(img, patch_size=14):
    """
    自动将图片resize到高宽均为patch_size的倍数（直接缩放，不保持原比例），并保证输出为3通道彩色图像。
    Args:
        img: numpy数组 (H,W,3)、(H,W,1) 或 (H,W)
        patch_size: int
    Returns:
        resized_img: numpy数组 (new_h, new_w, 3)
    """
    import cv2
    import numpy as np
    # 转为3通道彩色
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        pass
    else:
        raise ValueError(f"不支持的图片shape: {img.shape}")
    h, w = img.shape[:2]
    new_h = ((h + patch_size - 1) // patch_size) * patch_size
    new_w = ((w + patch_size - 1) // patch_size) * patch_size
    if (h, w) == (new_h, new_w):
        return img
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized


def extract_feature_vector(
    img_path,
    dino_net,
    patch_h,
    patch_w,
    feat_dim,
    num,
    device='cuda',
    show_heatmap=False,
    save_outputs=True,
):
    import os
    import torchvision.transforms as T
    from PIL import Image

    transform = T.Compose([
        T.Resize((patch_h * 14, patch_w * 14)),
        T.CenterCrop((patch_h * 14, patch_w * 14)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    img = Image.open(img_path).convert('RGB')
    imgs_tensor = torch.zeros(1, 3, patch_h * 14, patch_w * 14).to(device)
    imgs_tensor[0] = transform(img)[:3]

    with torch.no_grad():
        features_dict = dino_net.model.forward_features(imgs_tensor, num)
        features = features_dict['x_norm_patchtokens']

    features = features.reshape(1 * patch_h * patch_w, feat_dim).cpu().numpy()
    reduced_features = DinoNet.pca_reduce(features, out_dim=3)
    reshaped = reduced_features.reshape(16, 16, 3)

    heatmap = DinoNet.feature_to_heatmap(reshaped, method='mean')
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        orig_img = np.array(Image.open(img_path).convert('RGB'))
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)

    heatmap_resized = cv2.resize(
        heatmap,
        (orig_img.shape[1], orig_img.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    overlay = DinoNet.overlay_heatmap_on_image(orig_img, heatmap_resized, alpha=0.5)

    print('overlay dtype:', overlay.dtype, 'shape:', overlay.shape, 'min:', overlay.min(), 'max:', overlay.max())

    if save_outputs:
        target_dir = os.path.join(os.path.dirname(__file__), "outputs", "heatmaps")
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        overlay_path = os.path.join(target_dir, 'heatmap_overlay_' + str(num) + os.path.basename(img_path))
        overlay_with_colorbar_path = os.path.join(
            target_dir,
            'heatmap_overlay_colorbar_' + str(num) + os.path.basename(img_path),
        )
        print(overlay_path)
        cv2.imwrite(overlay_path, overlay)
        DinoNet.save_overlay_with_colorbar(
            overlay_bgr=overlay,
            save_path=overlay_with_colorbar_path,
            title='Feature Heatmap Overlay',
            cmap='jet',
        )

    if show_heatmap:
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Feature Heatmap Overlay')
        plt.show()

    pooled = torch.nn.functional.avg_pool2d(
        torch.from_numpy(reshaped.transpose(2, 0, 1)[None]),
        kernel_size=4,
        stride=4,
    )
    pooled_np = pooled.squeeze(0).permute(1, 2, 0).numpy()
    flattened = pooled_np.flatten()
    print(flattened.shape)
    return flattened

if __name__ == "__main__":
    import torch
    import torchvision.transforms as T
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from sklearn.decomposition import PCA
    import matplotlib
    import os

    # 参数设置
    patch_h = 16
    patch_w = 16
    feat_dim = 1024 # vitl14
    base_dir = os.path.dirname(__file__)
    ckpt_path = os.path.join(base_dir, "weights", "dinov2_vitl14_pretrain.pth")
    img_path1 = os.path.join(base_dir, "examples", "image1.jpg")
    img_path2 = os.path.join(base_dir, "examples", "image2.jpg")

    # 加载模型
    dino_net = DinoNet(cpt_path=ckpt_path, feature_layer=23, load_ckpt=True).cuda().eval()

    # 提取两张图片的特征向量

    vec1 = extract_feature_vector(
        img_path1,
        dino_net,
        patch_h,
        patch_w,
        feat_dim,
        num=24,
        show_heatmap=True,
        save_outputs=False,
    )
    # vec2 = extract_feature_vector(img_path2, dino_net, patch_h, patch_w, feat_dim,num=24)

    # 归一化
    # from sklearn.preprocessing import normalize
    # vec1 = normalize(vec1.reshape(1, -1))[0]
    # vec2 = normalize(vec2.reshape(1, -1))[0]

    # # 计算余弦相似度
    # from sklearn.metrics.pairwise import cosine_similarity
    # sim = cosine_similarity([vec1], [vec2])[0][0]
    # print(f"两张图片的余弦相似度: {sim:.4f}")
