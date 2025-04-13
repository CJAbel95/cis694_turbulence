#
# Filename: cutout_jhtdb.py
#
# Description: Python source corresponding to Jupyter notebook
#               here: https://github.com/idies/pyJHTDB/blob/master/examples/Use_JHTDB_in_windows.ipynb
#
#
# by Christopher J. Abel
#
# Revision History
# ----------------
#   04/13/2025 -- Original
#
# ------------------------------------------------------------
import zeep
import numpy as np
import struct
import base64


def main():
    client = zeep.Client('http://turbulence.pha.jhu.edu/service/turbulence.asmx?WSDL')
    ArrayOfFloat = client.get_type('ns0:ArrayOfFloat')
    ArrayOfArrayOfFloat = client.get_type('ns0:ArrayOfArrayOfFloat')
    SpatialInterpolation = client.get_type('ns0:SpatialInterpolation')
    TemporalInterpolation = client.get_type('ns0:TemporalInterpolation')

    token = "edu.jhu.pha.turbulence.testing-201406"  # replace with your own token

    nnp = 5  # number of points
    # points = np.random.rand(nnp, 3)
    x = np.linspace(0, 0.2, nnp)
    y = np.linspace(0, 0.2, nnp)
    z = np.full((5,), 0.1)
    points = np.column_stack((x, y, z))
    print(points.shape)

    # convert to JHTDB structures
    x_coor = ArrayOfFloat(points[:, 0].tolist())
    y_coor = ArrayOfFloat(points[:, 1].tolist())
    z_coor = ArrayOfFloat(points[:, 2].tolist())
    point = ArrayOfArrayOfFloat([x_coor, y_coor, z_coor]);

    print(points)

    Function_name = "GetVelocity"
    time = 0.6
    number_of_component = 3  # change this based on function_name, see http://turbulence.pha.jhu.edu/webquery/query.aspx
    result = client.service.GetData_Python(Function_name, token, "isotropic1024coarse", 0.6,
                                           SpatialInterpolation("Lag4"), TemporalInterpolation("None"), point)
    result = np.array(result).reshape((-1, number_of_component))

    print(result)

    Function_name = "GetPosition"
    startTime = 0.1
    endTime = 0.2
    dt = 0.02
    number_of_component = 3  # change this based on function_name, see http://turbulence.pha.jhu.edu/webquery/query.aspx

    position = client.service.GetPosition_Python(Function_name, token, "isotropic1024coarse", startTime, endTime, dt,
                                               SpatialInterpolation("None"), point)
    position = np.array(position).reshape((-1, number_of_component))
    print(position)

    ###############

    field = "u"
    timestep = 1
    x_start = 1
    y_start = 1
    z_start = 1
    x_end = 2
    y_end = 5
    z_end = 8
    x_step = 1
    y_step = 1
    z_step = 1
    filter_width = 0

    result = client.service.GetAnyCutoutWeb(token, "isotropic1024coarse", field, timestep,
                                            x_start, y_start, z_start, x_end, y_end, z_end,
                                            x_step, y_step, z_step, filter_width,
                                            "")  # put empty string for the last parameter

    # transfer base64 format to numpy
    number_of_component = 3  # change this based on the field
    nx = len(range(x_start, x_end + 1, x_step))
    ny = len(range(y_start, y_end + 1, y_step))
    nz = len(range(z_start, z_end + 1, z_step))
    base64_len = int(nx * ny * nz * number_of_component)
    base64_format = '<' + str(base64_len) + 'f'

    result = struct.unpack(base64_format, result)
    result = np.array(result).reshape((nz, ny, nx, number_of_component))
    print(result.shape)  # see the shape of the result and compare it with nx, ny, nz and number of component


if __name__ == '__main__':
    main()
