import os


def pmx_to_obj(mmd, geometry, vertices):
    # starting line
    output = "mtllib material.mtl\n\n"

    # vertex
    for i in range(0, vertices.shape[0]):
        vertex = vertices[i, :]
        output += "v {:f} {:f} {:f}\n".format(vertex[0], vertex[1], vertex[2])

    # uv
    for i in range(0, geometry.uvs.shape[0]):
        uv = geometry.uvs[i, :]
        output += "vt {:f} {:f}\n".format(uv[0], uv[1])

    # normal
    for i in range(0, geometry.normals.shape[0]):
        normal = geometry.normals[i, :]
        output += "vn {:f} {:f} {:f}\n".format(normal[0], normal[1], normal[2])

    face_start = 0
    # texture and face
    for mat in mmd.materials:
        output += "o {:s}\n".format(mat.name)
        output += "usemtl {:s}\n".format(mat.name)

        for j in range(face_start, face_start + mat.vertex_count // 3):
            face_line = []
            for k in range(0, 3):
                face_index = geometry.faces[j][k] + 1
                face_line.append("{:d}/{:d}/{:d}".format(face_index, face_index, face_index))
            output += "f " + " ".join(face_line) + "\n"
        face_start += mat.vertex_count // 3
    return output


def pmx_to_mtl(pmx):
    mtl_output = ""
    mat_dict = dict()

    for mat in pmx.materials:
        assert len(mat.name) > 0
        texture = pmx.textures[mat.texture_index]
        texture_name = texture.replace("\\", "/")
        texture_basename = os.path.basename(texture_name)

        mtl_output += f"newmtl {mat.name}\n"
        mtl_output += "Ns 10.0000\n"
        mtl_output += "Ni 1.5000\n"
        mtl_output += "d 1.0000\n"
        mtl_output += "Tr 0.0000\n"
        mtl_output += "Tf 1.0000 1.0000 1.0000\n"
        mtl_output += "illum 2\n"
        mtl_output += f"Ka {mat.diffuse_color.r:5f} {mat.diffuse_color.g:5f} {mat.diffuse_color.b:5f}\n"
        mtl_output += f"Kd {mat.diffuse_color.r:5f} {mat.diffuse_color.g:5f} {mat.diffuse_color.b:5f}\n"
        mtl_output += "Ks 0.0000 0.0000 0.0000\n"
        mtl_output += "Ke 0.0000 0.0000 0.0000\n"

        mtl_output += f"map_Ka {texture_basename}\n"
        mtl_output += f"map_Kd {texture_basename}\n"
        mat_dict[texture_basename] = texture_name

    return mat_dict, mtl_output
