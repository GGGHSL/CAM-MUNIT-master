from torch import nn
from models.networks import MLP, StyleEncoder, ContentEncoder, Decoder


class Generator(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params, main_device=None):
        super(Generator, self).__init__()
        dim = params['dim']  # 64
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        # n_upsample = params['n_upsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']

        # style encoder
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='adain', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, images):
        # reconstruct an image
        (content_cam, content_cam_logit), style_fake = self.encode(images)
        images_recon = self.decode(content_cam, style_fake)
        return images_recon, content_cam_logit

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content_cam, content_cam_logit = self.enc_content(images)
        return (content_cam, content_cam_logit), style_fake

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)  # assign self.dec's params
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in models
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the models
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

