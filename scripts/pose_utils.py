num_point_ver = 6
import math
import numpy as np

def genCameraPosition(look_at):
    quat_list = []
    rot_list = []
    trans_list = []
    position_list = []
    
    # alpha: 
    alpha = 0
    alpha_delta = (2 * math.pi) / num_point_ver
    for i in range(num_point_ver):
        alpha = alpha + alpha_delta
        flag_x = 1
        flag_y = 1
        alpha1 = alpha
        if alpha > math.pi/2 and alpha <= math.pi: 
            alpha1 = math.pi - alpha
            flag_x = -1
            flag_y = 1
        elif alpha > math.pi and alpha <= math.pi*(3/2):
            alpha1 = alpha - math.pi
            flag_x = -1
            flag_y = -1
        elif alpha > math.pi*(3/2):
            alpha1 = math.pi*2 - alpha
            flag_x = 1
            flag_y = -1
    
        beta = beta_range[0]
        beta_delta = (beta_range[1]-beta_range[0])/(num_point_hor-1)
        for j in range(num_point_hor):
            if j != 0:
                beta = beta + beta_delta

            x = flag_x * (r * math.sin(beta)) * math.cos(alpha1)
            y = flag_y * (r * math.sin(beta)) * math.sin(alpha1)
            z = r * math.cos(beta)
            position = np.array([x, y, z]) + look_at
            look_at = look_at
            up = np.array([0, 0, 1])

            vectorZ = - (look_at - position)/np.linalg.norm(look_at - position)
            vectorX = np.cross(up, vectorZ)/np.linalg.norm(np.cross(up, vectorZ))
            vectorY = np.cross(vectorZ, vectorX)/np.linalg.norm(np.cross(vectorX, vectorZ))

            # points in camera coordinates
            pointSensor= np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])

            # points in world coordinates 
            pointWorld = np.array([position,
                                   position + vectorX,
                                   position + vectorY * 2,
                                   position + vectorZ * 3])

            resR, resT = getRTFromAToB(pointSensor, pointWorld)
            resQ = quaternionFromRotMat(resR)

            quat_list.append(resQ)
            rot_list.append(resR)
            trans_list.append(resT)
            position_list.append(position)
    return quat_list, trans_list, rot_list 