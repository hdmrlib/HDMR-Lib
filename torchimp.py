import torch

def calculate_g0_torch(G, n1, n2, n3):
    s1 = torch.ones(n1, 1,dtype=torch.float64)
    s2 = torch.ones(n2, 1,dtype=torch.float64)
    s3 = torch.ones(n3, 1,dtype=torch.float64)
    w1,w2,w3 = 1/n1, 1/n2, 1/n3
    g0=torch.tensordot(torch.tensordot(torch.tensordot(G, s1, dims=([0], [0])), 
                                s2, dims=([0], [0])),s3,dims=([0], [0])) * w1 * w2*w3    
    return g0

def calculate_gi_torch(G, g0, n1, n2, n3):
    s1 = torch.ones(n1, 1,dtype=torch.float64)
    s2 = torch.ones(n2, 1,dtype=torch.float64)
    s3 = torch.ones(n3, 1,dtype=torch.float64)
    w1,w2,w3 = 1/n1, 1/n2, 1/n3
    g1=(torch.tensordot(torch.tensordot(G, s2, dims=([1], [0])), s3, dims=([1], [0])) * w2 * w3).reshape(n1,1)-(g0[0][0][0]*s1)
    g2=(torch.tensordot(torch.tensordot(G, s1, dims=([0], [0])), s3, dims=([1], [0])) * w1 * w3).reshape(n2,1)-(g0[0][0][0]*s2)
    g3=(torch.tensordot(torch.tensordot(G, s1, dims=([0], [0])), s2, dims=([0], [0])) * w1 * w2).reshape(n3,1)-(g0[0][0][0]*s3)
    return g1,g2,g3

def calculate_gij_torch(G, g0, g1, g2, g3, n1, n2, n3):
    s1 = torch.ones(n1, 1,dtype=torch.float64)
    s2 = torch.ones(n2, 1,dtype=torch.float64)
    s3 = torch.ones(n3, 1,dtype=torch.float64)
    w1,w2,w3 = 1/n1, 1/n2, 1/n3
    g12=(torch.tensordot(G_torch, s3, dims=([2], [0]))* w3).reshape(n1,n2)-(g0[0][0][0]*s1*s2.T)-(g1*s2.T)-(s1*g2.T)
    g13=(torch.tensordot(G_torch, s2, dims=([1], [0]))* w2).reshape(n1,n3)-(g0[0][0][0]*s1*s3.T)-(g1*s3.T)-(s1*g3.T)
    g23=(torch.tensordot(G_torch, s1, dims=([0], [0]))* w1).reshape(n2,n3)-(g0[0][0][0]*s2*s3.T)-(g2*s3.T)-(s2*g3.T)
    return g12,g13,g23