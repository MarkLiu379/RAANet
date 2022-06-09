import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.attention import Attention


class aspp_block(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, att_name='CBAM'):
        super(aspp_block, self).__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=rate, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.att = Attention(dim_out, dim_out, att_name)

    def forward(self, x):
        x = self.branch(x)
        x = self.att(x)
        return x


class res_ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1, att_name='CBAM'):
        super(res_ASPP, self).__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, bias=False),
            nn.BatchNorm2d(dim_out),
        )
        self.att = Attention(dim_out, dim_out, att_name)

        self.aspp_3 = aspp_block(dim_in, dim_out, rate=3 * rate, att_name=att_name)
        self.aspp_6 = aspp_block(dim_in, dim_out, rate=6 * rate, att_name=att_name)
        self.aspp_12 = aspp_block(dim_in, dim_out, rate=12 * rate, att_name=att_name)
        self.aspp_18 = aspp_block(dim_in, dim_out, rate=18 * rate, att_name=att_name)
        self.aspp_24 = aspp_block(dim_in, dim_out, rate=24 * rate, att_name=att_name)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        feature = self.conv_res(x)
        feature = self.att(feature)

        aspp3_1 = self.aspp_3(x)
        aspp3 = aspp3_1 + feature

        aspp6_1 = self.aspp_6(x)
        aspp6 = aspp6_1 + feature

        aspp12_1 = self.aspp_12(x)
        aspp12 = aspp12_1 + feature

        aspp18_1 = self.aspp_18(x)
        aspp18 = aspp18_1 + feature

        aspp24_1 = self.aspp_24(x)
        aspp24 = aspp24_1 + feature

        feature = torch.cat([aspp3, aspp6, aspp12, aspp18, aspp24], dim=1)

        result = self.conv_cat(feature)

        # aspp3 = F.interpolate(aspp3, size=(512, 512), mode='bilinear', align_corners=False)
        # aspp6 = F.interpolate(aspp6, size=(512, 512), mode='bilinear', align_corners=False)
        # aspp12 = F.interpolate(aspp12, size=(512, 512), mode='bilinear', align_corners=False)
        # aspp18 = F.interpolate(aspp18, size=(512, 512), mode='bilinear', align_corners=False)
        # aspp24 = F.interpolate(aspp24, size=(512, 512), mode='bilinear', align_corners=False)
        # return result, aspp3, aspp6, aspp12, aspp18, aspp24
        return result


# -----------------------------------------#
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
# -----------------------------------------#
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)

        # conv1x1 = F.interpolate(conv1x1, size=(512, 512), mode='bilinear', align_corners=False)
        # conv3x3_1 = F.interpolate(conv3x3_1, size=(512, 512), mode='bilinear', align_corners=False)
        # conv3x3_2 = F.interpolate(conv3x3_2, size=(512, 512), mode='bilinear', align_corners=False)
        # conv3x3_3 = F.interpolate(conv3x3_3, size=(512, 512), mode='bilinear', align_corners=False)
        # global_feature = F.interpolate(global_feature, size=(512, 512), mode='bilinear', align_corners=False)
        # return result, conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature
        return result


class RAANet(nn.Module):
    def __init__(self, num_classes, backbone="xception", att_name='CBAM', aspp_name='res', pretrained=True,
                 downsample_factor=16):
        super(RAANet, self).__init__()
        self.att_name = att_name
        if backbone == "xception":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            # ----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
            # ----------------------------------#
            #   注意力层
            # ----------------------------------#
            if self.att_name != 'null':
                self.att = Attention(2048, 2048, att_name)
                self.att_1 = Attention(256, 256, att_name)
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        # -----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        # -----------------------------------------#
        if aspp_name == 'res':
            self.aspp = res_ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor, att_name=att_name)
        else:
            assert aspp_name == 'null'
            self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)
        # ----------------------------------#
        #   浅层特征边
        # ----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1)
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        # -----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        # -----------------------------------------#
        low_level_features, x = self.backbone(x)

        if self.att_name != 'null':
            low_level_features = self.att_1(low_level_features)
            atten = self.att(x)
            x = self.aspp(atten)

            # low_vis = F.interpolate(low_level_features, size=(512, 512), mode='bilinear', align_corners=False)
            # x, aspp1, aspp2, aspp3, aspp4, aspp5 = self.aspp(atten)
            # vis_total = x
            # vis_total = F.interpolate(vis_total, size=(512, 512), mode='bilinear', align_corners=False)
        else:
            x = self.aspp(x)

            # low_vis = F.interpolate(low_level_features, size=(512, 512), mode='bilinear', align_corners=False)
            # x, aspp1, aspp2, aspp3, aspp4, aspp5 = self.aspp(x)
            # vis_total = x
            # vis_total = F.interpolate(vis_total, size=(512, 512), mode='bilinear', align_corners=False)

        low_level_features = self.shortcut_conv(low_level_features)

        # -----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        # -----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        # return x, low_vis, aspp1, aspp2, aspp3, aspp4, aspp5, vis_total
        return x
