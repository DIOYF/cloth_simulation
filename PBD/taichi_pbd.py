import taichi as ti

ti.init(arch=ti.cuda)

# ==============create cloth mesh info ==============
n = 128 # particle number in col and row
Pos = ti.Vector.field(3, dtype=float, shape=(n, n))
Vel = ti.Vector.field(3, dtype=float, shape=(n, n))
Triangles = ti.field(int, shape = (n-1)*(n-1)*6)  # edge info about spring
colors = ti.Vector.field(3, dtype=float, shape=n * n)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)

#initial_edge and indices info. For convenience, spring original lenght is quad_size or quad_size * 1.414
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
dt = 0.03 # 0.001

gravity = ti.Vector([0, -0.0098, 0])


ball_radius = 0.2
ball_center = ti.Vector.field(3, dtype=float, shape=(1,))


# just need fix this function to use different cloth simulation methods
@ti.kernel
def add_gravity():
    # add gravity
    for i in ti.grouped(Pos):
        if i[1] == 0 and (i[0] == 0 or i[0] == 127):
            pass
        else:
            Vel[i] += gravity * dt
            Pos[i] += Vel[i] * dt


spring_offsets = []
for i in range(-1, 2):
    for j in range(-1, 2):
        if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
            spring_offsets.append(ti.Vector([i, j]))
sum_x = ti.Vector.field(3, dtype=float, shape=(n, n))
sum_n = ti.field(float, shape=(n, n))
@ti.kernel
def strain_limiting():
    for i in ti.grouped(Pos):
        sum_x[i] = [0, 0, 0]
        sum_n[i] = 0
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                direction = (Pos[i] - Pos[j]).normalized()
                sum_n[i] += 1.0
                sum_x[i] += (Pos[i] + Pos[j] + quad_size * float(i-j).norm() * direction) / 2

        ti.sync()
        if i[1] == 0 and (i[0] == 0 or i[0] == 127):
            pass
        else:
            Vel[i] += 1/dt * ((0.4 * Pos[i] + sum_x[i]) / (0.4 + sum_n[i]) - Pos[i])
            Pos[i] = (0.4 * Pos[i] + sum_x[i]) / (0.4 + sum_n[i])


@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = Pos[i, j]


@ti.kernel
def collision_handing_with_ball():
    # how to react with moving ball ? no overshooting ,but explict euler will overshooting
    for i in ti.grouped(Pos):
        if (Pos[i] - ball_center[0]).norm() < ball_radius:
            direction = (Pos[i] - ball_center[0]).normalized()
            Vel[i] = Vel[i] + 1/dt * (ball_center[0] - Pos[i] + ball_radius * direction)
            Pos[i] = ball_center[0] + ball_radius * direction


def main():
    window = ti.ui.Window("Taichi Cloth Simulation with explict euler ", (800, 600), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((0, 0, 0))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    initial_spring_mesh()
    current_time = 0.0
    num_iters = 64
    while window.running:
        current_time += dt
        # when ball move faster, what happen?
        ball_center[0] = [0.0, 0, 0.5 * ti.sin(current_time*0.1) - 0.5]

        add_gravity()
        for i in range(num_iters):
            strain_limiting()

        collision_handing_with_ball()
        update_vertices()
        # you can try this direction to view, what happen?
        # camera.position(1.1, 0.0, 1.1)
        # camera.lookat(-0.5, 0.0, -0.5)
        camera.position(-1.1, 0.5, -1.1)
        camera.lookat(0.5, 0.0, 0.3)

        scene.set_camera(camera)
        # light set
        scene.point_light(pos=(0, 1, 2), color=(0.5, 0.5, 0.5))
        scene.point_light(pos=(-1, -3, -4), color=(0.5, 0.5, 0.5))
        # cloth mesh set
        scene.mesh(vertices, indices=Triangles, per_vertex_color=colors, two_sided=True)
        # scene particles
        scene.particles(ball_center, radius=ball_radius*0.95, color=(122/255, 144/255, 188/255)) # smaller ball to escape inter-model

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
