import torch
import torch.nn as nn

'''
CBN (Conditional Batch Normalization layer)
    uses an MLP to predict the beta and gamma parameters in the batch norm equation
    Reference : https://papers.nips.cc/paper/7237-modulating-early-visual-processing-by-language.pdf
'''
class BN(nn.Module):

    def __init__(self, out_size, batch_size, channels, height, width, use_betas=True, use_gammas=True, eps=1.0e-5):
        super(BN, self).__init__()

        self.out_size = out_size # output of the MLP - for each channel
        self.use_betas = use_betas
        self.use_gammas = use_gammas

        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width

        # beta and gamma parameters for each channel - defined as trainable parameters
        self.betas = nn.Parameter(torch.zeros(self.batch_size, self.channels).cuda())
        self.gammas = nn.Parameter(torch.ones(self.batch_size, self.channels).cuda())
        self.eps = eps


        # initialize weights using Xavier initialization and biases with constant value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0.1)

    '''
    Computer Normalized feature map with the updated beta and gamma values
    Arguments:
        feature : feature map from the previous layer
        stat_emb : stat embedding of the question
    Returns:
        out : beta and gamma normalized feature map
        stat_emb : stat embedding of the question (unchanged)
    Note : stat_emb needs to be returned since CBN is defined within nn.Sequential
           and subsequent CBN layers will also require stat question embeddings
    '''
    def forward(self, feature, stat_emb):
       
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
        feature_normalized = (feature-batch_mean)/torch.sqrt(batch_var+self.eps)

        # get the normalized feature map with the updated beta and gamma values
        out = torch.mul(feature_normalized, gammas_expanded) + betas_expanded

        return out, stat_emb

'''
# testing code
if __name__ == '__main__':
    torch.cuda.set_device(int(sys.argv[1]))
    model = CBN(512, 256)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            print 'found anomaly'
        if isinstance(m, nn.Linear):
            print 'found correct'
'''