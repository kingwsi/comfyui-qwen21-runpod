"""API graphs for the pinned Qwen 2.1 / GGUF runtime (no torch import)."""
import secrets

SIGMAS = '1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0'
MAX_IMAGES = 3
BFS_FILENAME = 'bfs_head_v1.1_qwen_2.1.safetensors'
BFS_PROMPT = ('head_swap: start with <image1> as the base image, keeping its lighting, '
              'environment, and background. remove the head from <image1> completely '
              'and replace it with the head from <image2>, strictly preserving the hair, '
              'eye color, nose structure from <image2>. copy the direction of the eye, '
              'head rotation, micro expressions from <image1>, high quality, sharp details, 4k')
TEMPLATES = {
    'text': dict(name='文生图', description='用文字描述，生成一张新图片', min_images=0, max_images=0,
                 placeholder='例如：一张简约的山海旅行海报，标题是「向海而生」，清新的蓝绿色调。', default_prompt=''),
    'edit': dict(name='通用编辑', description='上传一张图，描述需要修改的地方', min_images=1, max_images=1,
                 placeholder='例如：将衣服改成深蓝色，其他保持不变。', default_prompt=''),
    'reference': dict(name='多图参考', description='第一张为主图，其余图片提供参考', min_images=2, max_images=3,
                 placeholder='说明每张参考图的用途，例如：参考图 2 的背景、图 3 的光线，保留图 1 的人物。',
                 default_prompt='编辑第 1 张主图，参考第 2 张图的背景与配色；如有第 3 张图，参考其光线与氛围。保持主图主体和构图不变，输出一张自然协调的完整图片，不要拼图。'),
    'faceswap': dict(name='换脸 / 换头', description='两张图片，替换人物头部与身份', min_images=2, max_images=2,
                    required_variant='turbo', image_labels=['目标图片', '参考头像'],
                    image_hint='图 1 保留身体、姿势和场景；图 2 提供五官与发型。会替换整个头部，也可能改变发型。',
                    placeholder='可补充头部融合、表情或细节要求。',
                    default_prompt='使用参考头像的身份与发型，保留目标图片的头部角度、视线和表情，保持身体、服装与背景不变，头颈衔接自然。'),
}


def templates_for_variant(variant):
    return {key: spec for key, spec in TEMPLATES.items()
            if spec.get('required_variant', variant) == variant}


def template_prompt(template, text, image_count):
    if template not in TEMPLATES:
        raise ValueError('请选择有效模板。')
    spec = TEMPLATES[template]
    if not spec['min_images'] <= image_count <= spec['max_images']:
        raise ValueError(f"{spec['name']}需要 {spec['min_images']}–{spec['max_images']} 张图片。")
    if not isinstance(text, str) or not text.strip() or len(text) > 4000:
        raise ValueError('请输入 1–4000 字的生成要求。')
    prefix = {
        'text': '',
        'edit': '根据要求编辑原图。保留未要求修改的内容。\n',
        'reference': '第 1 张为主图，第 2 张及之后为参考图。按要求将参考元素应用于主图，生成一张完整图片，不要拼图。\n',
        'faceswap': BFS_PROMPT + '\nAdditional instructions: ',
    }[template]
    return prefix + text.strip()


def build_graph(prompt, images, variant='base', seed=None, template=None):
    if variant not in ('base', 'turbo') or len(images) > MAX_IMAGES:
        raise ValueError('不支持的模型或参考图数量。')
    if template == 'faceswap' and (variant != 'turbo' or len(images) != 2):
        raise ValueError('换脸 / 换头模板目前仅支持 Turbo，需要目标图片和参考头像各一张。')
    seed = secrets.randbelow(2**53) if seed is None else seed
    def node(kind, **inputs):
        return {'class_type': kind, 'inputs': inputs}
    turbo = variant == 'turbo'
    graph = {
        '1': node('UnetLoaderGGUF', unet_name='Qwen-Image-2.1-Turbo-Q8_0.gguf' if turbo else 'qwen-image-2.1-UC-Q8_0.gguf'),
        '2': node('CLIPLoaderGGUF', clip_name='Qwen-Image-2.1-Turbo-Uncensored-Q8_0.gguf', type='qwen_image') if turbo else node('CLIPLoader', clip_name='qwen3vl_8b_bf16.safetensors', type='qwen_image', device='default'),
        '3': node('VAELoader', vae_name='qwen_image_2.1_vae_bf16.safetensors'),
        '4': node('TextEncodeQwenImage21', clip=['2', 0], vae=['3', 0], prompt=prompt, negative_prompt='', resolution=0 if images else 1024),
        '5': node('QwenImage21Cache', model=['1', 0], device='auto', dtype='default'),
        '7': node('VAEDecode', samples=['6', 0], vae=['3', 0]),
        '8': node('SplitImageWithAlpha', image=['7', 0]),
        '9': node('SaveImage', images=['8', 0], filename_prefix='qwen_studio/image'),
    }
    if template == 'faceswap':
        graph['12'] = node('LoraLoaderModelOnly', model=['1', 0], lora_name=BFS_FILENAME, strength_model=1.0)
        graph['5']['inputs']['model'] = ['12', 0]
    if turbo:
        graph.update({
            '6': node('SamplerCustom', model=['5', 0], add_noise=True, noise_seed=seed, cfg=1.0, positive=['4', 0], negative=['4', 1], sampler=['10', 0], sigmas=['11', 0], latent_image=['4', 2]),
            '10': node('KSamplerSelect', sampler_name='euler'),
            '11': node('ManualSigmas', sigmas=SIGMAS),
        })
    else:
        graph['6'] = node('KSampler', model=['5', 0], seed=seed, steps=25, cfg=1.0, sampler_name='euler', scheduler='simple', positive=['4', 0], negative=['4', 1], latent_image=['4', 2], denoise=1.0)
    for i, image in enumerate(images, 1):
        load, scale = str(20 + 2*i), str(21 + 2*i)
        graph[load] = node('LoadImage', image=image)
        graph[scale] = node('ImageScaleToMaxDimension', image=[load, 0], upscale_method='lanczos', largest_size=1024)
        graph['4']['inputs'][f'images.image_{i}'] = [scale, 0]
    return graph
