import torch
import torch.nn as nn

'''
CBN (Conditional Batch Normalization layer)
    uses an MLP to predict the beta and gamma parameters in the batch norm equation
    Reference : https://papers.nips.cc/paper/7237-modulating-early-visual-processing-by-language.pdf
'''
class BN(nn.Module):

    def __init__(self, channels):
        super(BN, self).__init__()

        self.channels = channels

        # beta and gamma parameters for each channel - defined as trainable parameters
        self.betas = nn.Parameter(torch.zeros(32, self.channels).cuda())
        self.gammas = nn.Parameter(torch.ones(32, self.channels).cuda())


    def forward(self, feature):
        self.batch_size, self.channels, self.height, self.width = feature.data.shape



        betas_cloned = self.betas.clone()
        gammas_cloned = self.gammas.clone()



        # get the mean and variance for the batch norm layer
        batch_mean = torch.mean(feature)
        batch_var = torch.var(feature)

        # extend the betas and gammas of each channel across the height and width of feature map
        betas_expanded = torch.stack([betas_cloned]*self.height, dim=2)
        betas_expanded = torch.stack([betas_expanded]*self.width, dim=3)

        gammas_expanded = torch.stack([gammas_cloned]*self.height, dim=2)
        gammas_expanded = torch.stack([gammas_expanded]*self.width, dim=3)

        # normalize the feature map
        feature_normalized = (feature-batch_mean)/torch.sqrt(batch_var+1.0e-5)

        # get the normalized feature map with the updated beta and gamma values
        out = torch.mul(feature_normalized, gammas_expanded) + betas_expanded

        return out


class FN(nn.Module):

    def __init__(self, channels):
        super(FN, self).__init__()

        self.channels = channels

        # beta and gamma parameters for each channel - defined as trainable parameters
        #self.betas = nn.Parameter(torch.zeros(32, self.channels).cuda())
        #self.gammas = nn.Parameter(torch.ones(32, self.channels).cuda())


    def forward(self, feature):
        self.batch_size, self.channels, self.height, self.width = feature.data.shape



        #betas_cloned = self.betas.clone()
        #gammas_cloned = self.gammas.clone()



        # get the mean and variance for the batch norm layer
        batch_mean = torch.mean(feature)
        batch_var = torch.var(feature)

        # extend the betas and gammas of each channel across the height and width of feature map
        #betas_expanded = torch.stack([betas_cloned]*self.height, dim=2)
        #betas_expanded = torch.stack([betas_expanded]*self.width, dim=3)

        #gammas_expanded = torch.stack([gammas_cloned]*self.height, dim=2)
        #gammas_expanded = torch.stack([gammas_expanded]*self.width, dim=3)

        # normalize the feature map
        feature_normalized = (feature-batch_mean)/torch.sqrt(batch_var+1.0e-5)

        # get the normalized feature map with the updated beta and gamma values
        #out = torch.mul(feature_normalized, gammas_expanded) + betas_expanded

        return feature_normalized