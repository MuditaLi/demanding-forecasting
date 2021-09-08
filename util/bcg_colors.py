BCG_GREY = '#F2F2F2'
BCG_LIGHT_GREEN = '#28BB74'
BCG_DARK_GREEN = '#3E864F'


def paint_plt_bcg_ready(plt) -> None:
    plt.gca().set_facecolor(BCG_GREY)
    plt.gcf().patch.set_facecolor(BCG_GREY)
    plt.gcf().set_size_inches(8, 6)
