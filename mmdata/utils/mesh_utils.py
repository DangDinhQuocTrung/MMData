import numpy as np
import trimesh


def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, "w")
    for v in verts:
        file.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write("f %d %d %d\n" % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()


def load_obj_mesh(mesh_file, key_name=None, with_normal=False, with_texture=False):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []
    get_faces = False

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file

    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith("#"):
            continue

        values = line.split()
        if not values:
            continue

        if key_name is None:
            get_faces = True
        if values[0] == "v":
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == "vn":
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == "vt":
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == "o":
            if key_name is None:
                get_faces = True
            elif values[1] == key_name:
                get_faces = True
            else:
                get_faces = False
        elif values[0] == "f" and get_faces:
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split("/")[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split("/")[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split("/")[0]), values[1:4]))
                face_data.append(f)

            # deal with texture
            if len(values[1].split("/")) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split("/")[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split("/")[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split("/")[1]) != 0:
                    f = list(map(lambda x: int(x.split("/")[1]), values[1:4]))
                    face_uv_data.append(f)

            # deal with normal
            if len(values[1].split("/")) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split("/")[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split("/")[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split("/")[2]) != 0:
                    f = list(map(lambda x: int(x.split("/")[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        norms = np.array(norm_data)
        norms = normalize_v3(norms)
        face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    return vertices, faces


def normalize_v3(arr):
    """ Normalize a numpy array of 3 component vectors shape=(n,3) """
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would contribute to every vertex, so we need to normalize afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)
    return norm


def compute_tangent(vertices, faces, normals, uvs, faceuvs):
    # NOTE: this could be numerically unstable around [0,0,1]
    # but other current solutions are pretty freaky somehow
    c1 = np.cross(normals, np.array([0, 1, 0.0]))
    tan = c1
    normalize_v3(tan)
    btan = np.cross(normals, tan)
    return tan, btan


def merge_scene_to_mesh(scene):
    if isinstance(scene, trimesh.Scene):
        if len(scene.geometry) == 0:
            mesh = None
        else:
            # we lose texture information here
            geometry_parts = [tm for tm in list(scene.geometry.values())]
            mesh = trimesh.util.concatenate(geometry_parts)
    else:
        assert isinstance(scene, trimesh.Trimesh)
        mesh = scene
    return mesh


def display_mesh(mesh):
    if isinstance(mesh, trimesh.Scene):
        scene = mesh
        for geometry in scene.geometry.values():
            geometry.vertices[..., 2] = -geometry.vertices[..., 2]
        scene.show()
    else:
        mesh.vertices[..., 2] = -mesh.vertices[..., 2]
        mesh.show()
    return
