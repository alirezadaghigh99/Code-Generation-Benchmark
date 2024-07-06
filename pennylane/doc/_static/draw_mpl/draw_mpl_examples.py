def use_style(circuit):

    fig, ax = qml.draw_mpl(circuit, style="sketch")(1.2345, 1.2345)

    plt.savefig(folder / "sketch_style.png")
    plt.close()
    qml.drawer.use_style("black_white")

