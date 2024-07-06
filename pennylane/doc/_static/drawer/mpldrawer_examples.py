def CNOT(savefile="cnot.png"):
    drawer = MPLDrawer(n_wires=2, n_layers=2)

    drawer.CNOT(0, (0, 1))

    options = {"color": "indigo", "linewidth": 4}
    drawer.CNOT(1, (1, 0), options=options)
    plt.savefig(folder / savefile)
    plt.close()

def measure(savefile="measure.png"):
    drawer = MPLDrawer(n_wires=2, n_layers=1)
    drawer.measure(layer=0, wires=0)

    measure_box = {"facecolor": "white", "edgecolor": "indigo"}
    measure_lines = {"edgecolor": "indigo", "facecolor": "plum", "linewidth": 2}
    drawer.measure(layer=0, wires=1, box_options=measure_box, lines_options=measure_lines)

    plt.savefig(folder / savefile)
    plt.close()

def ctrl(savefile="ctrl.png"):
    drawer = MPLDrawer(n_wires=2, n_layers=3)

    drawer.ctrl(layer=0, wires=0, wires_target=1)
    drawer.ctrl(layer=1, wires=(0, 1), control_values=[0, 1])

    options = {"color": "indigo", "linewidth": 4}
    drawer.ctrl(layer=2, wires=(0, 1), control_values=[1, 0], options=options)

    plt.savefig(folder / savefile)
    plt.close()

