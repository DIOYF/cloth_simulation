import taichi as ti

ti.init(arch=ti.cuda)

# ==============create cloth mesh info ==============
n = 128 # particle number in col and row
Pos = ti.Vector.field(3, dtype=float, shape=(n, n))
Vel = ti.Vector.field(3, dtype=float, shape=(n, n))
Triangles = ti.field(int, shape = (n-1)*(n-1)*6)  # edge info about spring
colors = ti.Vector.field(3, dtype=float, shape= n * n)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)

#initial_edge and indices info. For convenience, spring original lenght is quad_size
quad_size = 1/n



@ti.kernel
def initial_spring_mesh():
    # initial each particle's position and velocity
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.01
    for i, j in Pos:
        Pos[i, j] = [i * quad_size - 0.5 + random_offset[0], 0.6, j * quad_size - 0.5 + random_offset[1]]
        Vel[i, j] = [0, 0, 0]

    # initial triangle mesh indices and its velocity
    for i, j in ti.ndrange(n-1, n-1):
        indice_id = i * (n-1) + j
        Triangles[indice_id * 6 + 0] = i * n + j
        Triangles[indice_id * 6 + 1] = (i+1) * n + j
        Triangles[indice_id * 6 + 2] = i * n + (j+1)
        Triangles[indice_id * 6 + 3] = (i+1) * n + j + 1
        Triangles[indice_id * 6 + 4] = i * n + (j + 1)
        Triangles[indice_id * 6 + 5] = (i + 1) * n + j

    # UV mapping , just initial color of corresponding vertices
    for i, j in ti.ndrange(n, n):
        if (i // 2 + j // 2) % 2 == 0:
            colors[i * n + j] = (192 / 255, 208 / 255, 157 / 255)
        else:
            colors[i * n + j] = (245 / 255, 251 / 255, 254 / 255)


# ==============set scene time info and force info ==============
dt = 1e-4 # 0.0004
substeps = int(1/120//dt)  # substep to escape explicit integration overshooting
gravity = ti.Vector([0, -9.8, 0])

ball_radius = 0.2
ball_center = ti.Vector.field(3, dtype=float, shape= (1,))


spring_k = 3e4  # stiffness factor
dashpot_damping = 1e4  # velocity damping loss
drag_damping = 1 # spring loss


spring_offsets = []
for i in range(-1, 2):
    for j in range(-1, 2):
        if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
            spring_offsets.append(ti.Vector([i, j]))


# just need fix this function to use different cloth simulation methods
@ti.kernel
def substep():
    for i in ti.grouped(Pos):
        force = gravity
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = Pos[i] - Pos[j]
                v_ij = Vel[i] - Vel[j]
                direction = x_ij.normalized()  # 获得力的方向
                current_dist = x_ij.norm()  # 距离，根据胡克定律获得力的大小
                original_dist = quad_size * float(i - j).norm()   # 质点初始距离
                # spring force
                force += -spring_k * direction * (current_dist / original_dist - 1)  # 1 means original_dist - original_dist

                # Dashpot damping
                force += -v_ij.dot(direction) * direction * dashpot_damping * quad_size
        if i[1] == 0 and (i[0] == 0 or i[0] == 127):
            pass
        else:
            Vel[i] += force * dt
            Vel[i] *= ti.exp(-drag_damping * dt)

            # collision detection: update position is cloth rush into
            offset_to_center = Pos[i] - ball_center[0]
            if offset_to_center.norm() <= ball_radius:
                normal = offset_to_center.normalized()
                Vel[i] -= min(Vel[i].dot(normal), 0) * normal # simply progress cloth collision , make v_n = 0
            Pos[i] += Vel[i] * dt




@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = Pos[i, j]


@ti.kernel
def interactive_ball():
    current_time = ti.random()
    ball_center[0] = [0.0, 0, 0.05 * ti.sin(current_time) - 0.5]


def main():
    window = ti.ui.Window("Taichi Cloth Simulation with explict euler ", (800, 600), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((0, 0, 0))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    initial_spring_mesh()
    current_time = 0.0

    while window.running:
        # you can try this ball move , and you will find that the ball rust into the cloth ,this
        # is because our cloth simulation collision process is naive
        # current_time += 0.01
        # ball_center[0] = [0.1 * ti.sin(current_time), 0.0, 0.1 * ti.sin(current_time) - 0.5]

        for i in range(substeps):
            substep()
        update_vertices()

        camera.position(1.1, 0.0, 1.1)
        camera.lookat(-0.5, 0.0, -0.5)
        scene.set_camera(camera)
        # light set
        scene.point_light(pos=(0, 1, 2), color=(0.5, 0.5, 0.5))
        scene.point_light(pos=(1, 5, 4), color=(0.5, 0.5, 0.5))
        # cloth mesh set
        scene.mesh(vertices, indices=Triangles, per_vertex_color=colors, two_sided=True)
        # scene particles
        scene.particles(ball_center, radius=ball_radius*0.95, color=(122/255, 144/255, 188/255)) # smaller ball to escape inter-model

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()


