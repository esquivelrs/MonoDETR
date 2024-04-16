#!/usr/bin/env python
# Made by: Jonathan Mikler
# Creation date: 2024-04-16

import torch

def to_float_tensor(input):

    data_type = type(input)

    if data_type != torch.Tensor:
        input = torch.tensor(input)
    
    return input.float()

def get_cuboid_verts_faces(box3d=None, R=None):
    """
    Computes vertices and faces from a 3D cuboid representation.
    Args:
        bbox3d (flexible): [[X Y Z W H L]]
        R (flexible): [np.array(3x3)]
    Returns:
        verts: the 3D vertices of the cuboid in camera space
        faces: the vertex indices per face
    """
    if box3d is None:
        box3d = [0, 0, 0, 1, 1, 1]

    # make sure types are correct
    box3d = to_float_tensor(box3d)
    
    if R is not None:
        R = to_float_tensor(R)

    squeeze = len(box3d.shape) == 1
    
    if squeeze:    
        box3d = box3d.unsqueeze(0)
        if R is not None:
            R = R.unsqueeze(0)
    
    n = len(box3d)

    x3d = box3d[:, 0].unsqueeze(1)
    y3d = box3d[:, 1].unsqueeze(1)
    z3d = box3d[:, 2].unsqueeze(1)
    w3d = box3d[:, 3].unsqueeze(1)
    h3d = box3d[:, 4].unsqueeze(1)
    l3d = box3d[:, 5].unsqueeze(1)

    '''
                    v4_____________________v5
                    /|                    /|
                   / |                   / |
                  /  |                  /  |
                 /___|_________________/   |
              v0|    |                 |v1 |
                |    |                 |   |
                |    |                 |   |
                |    |                 |   |
                |    |_________________|___|
                |   / v7               |   /v6
                |  /                   |  /
                | /                    | /
                |/_____________________|/
                v3                     v2
    '''

    verts = to_float_tensor(torch.zeros([n, 3, 8], device=box3d.device))

    # setup X
    verts[:, 0, [0, 3, 4, 7]] = -l3d / 2
    verts[:, 0, [1, 2, 5, 6]] = l3d / 2

    # setup Y
    verts[:, 1, [0, 1, 4, 5]] = -h3d / 2
    verts[:, 1, [2, 3, 6, 7]] = h3d / 2

    # setup Z
    verts[:, 2, [0, 1, 2, 3]] = -w3d / 2
    verts[:, 2, [4, 5, 6, 7]] = w3d / 2

    if R is not None:

        # rotate
        verts = R @ verts
    
    # translate
    verts[:, 0, :] += x3d
    verts[:, 1, :] += y3d
    verts[:, 2, :] += z3d

    verts = verts.transpose(1, 2)

    faces = torch.tensor([
        [0, 1, 2], # front TR
        [2, 3, 0], # front BL

        [1, 5, 6], # right TR
        [6, 2, 1], # right BL

        [4, 0, 3], # left TR
        [3, 7, 4], # left BL

        [5, 4, 7], # back TR
        [7, 6, 5], # back BL

        [4, 5, 1], # top TR
        [1, 0, 4], # top BL

        [3, 2, 6], # bottom TR
        [6, 7, 3], # bottom BL
    ]).float().unsqueeze(0).repeat([n, 1, 1])

    if squeeze:
        verts = verts.squeeze()
        faces = faces.squeeze()

    return verts, faces.to(verts.device)
