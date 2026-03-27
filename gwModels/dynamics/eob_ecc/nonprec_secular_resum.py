"""
Eccentricity-resummed 3PN evolution ODEs for aligned-spin eccentric binaries.
Numba JIT-compiled for performance.
"""

import math
from numba import njit
from ._common import (
    EULER_GAMMA, LOG2, LOG3, LOG5, PI,
    _zetadot_numba, integrate_dynamics,
)


@njit(cache=True, fastmath=True)
def _edot_resum_numba(e, x, nu, chi_S, chi_A, delta, kappa_S, kappa_A):
    e2 = e * e
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e4 * e4
    nu2 = nu * nu
    nu3 = nu2 * nu
    oome2 = 1.0 - e2
    sqrt_oome2 = math.sqrt(oome2)
    sqrt_x = math.sqrt(x)

    chiA2 = chi_A * chi_A
    chiS2 = chi_S * chi_S
    chiAS = chi_A * chi_S

    prefactor = -304.0 * e * x**4 * nu / (15.0 * oome2**2.5)

    # 0PN
    term_0pn = 1.0 + 121.0 * e2 / 304.0

    # 1.5PN tail
    tail_15pn = (PI * x**1.5
                 * (189120.0 + 286512.0 * e2 + 24217.0 * e4 - 98.0 * e6)
                 / (29184.0 * oome2**1.5))

    # 1.5PN SO
    so_15pn_num = (-16232.0 * delta * chi_A
                   + 8.0 * (-2029.0 + 1664.0 * nu) * chi_S
                   + e4 * (1869.0 * delta * chi_A
                           + (1869.0 - 276.0 * nu) * chi_S)
                   + e2 * (-248.0 * delta * chi_A
                           + (-248.0 + 7972.0 * nu) * chi_S))
    so_15pn = x**1.5 * so_15pn_num / (1824.0 * oome2**1.5)

    # 1PN
    pn1_num = (e4 * (94887.0 + 19768.0 * nu)
               + 8.0 * (20547.0 + 24556.0 * nu)
               + e2 * (464376.0 + 257124.0 * nu))
    term_1pn = -x * pn1_num / (51072.0 * oome2)

    # 2.5PN tail
    tail_25pn_num = (1372.0 * e8 * (-9.0 + nu)
                     + 192.0 * (263841.0 + 577888.0 * nu)
                     + e6 * (8626445.0 + 1850510.0 * nu)
                     + 25.0 * e4 * (9908007.0 + 4499656.0 * nu)
                     + e2 * (302170212.0 + 319380528.0 * nu))
    tail_25pn = -PI * x**2.5 * tail_25pn_num / (2451456.0 * oome2**2.5)

    # 2.5PN SO tail
    so_25pn_tail_num = (-2360832.0 * delta * chi_A
                        + 384.0 * (-6148.0 + 4937.0 * nu) * chi_S
                        + e6 * (132438.0 * delta * chi_A
                                + (132438.0 - 42287.0 * nu) * chi_S)
                        + e2 * (-1882704.0 * delta * chi_A
                                + 48.0 * (-39223.0 + 74954.0 * nu) * chi_S)
                        + e4 * (2251234.0 * delta * chi_A
                                + (2251234.0 + 191122.0 * nu) * chi_S))
    so_25pn_tail = PI * x**3 * so_25pn_tail_num / (43776.0 * oome2**3)

    # 2PN
    pn2_num = (-3.0 * e6 * (1056441.0 + 339608.0 * nu + 60256.0 * nu2)
               - 16.0 * (-765197.0 + 772695.0 * nu + 225792.0 * nu2)
               - 12.0 * e2 * (-949135.0 + 4137363.0 * nu + 987567.0 * nu2)
               - 2.0 * e4 * (11094859.0 + 13799931.0 * nu + 2536968.0 * nu2)
               + sqrt_oome2 * (1532160.0 * (-5.0 + 2.0 * nu)
                               + 609840.0 * e2 * (-5.0 + 2.0 * nu)))
    term_2pn = -x**2 * pn2_num / (612864.0 * oome2**2)

    # 2PN SS
    ss_2pn_num = (13472.0 * (delta * kappa_A + kappa_S
                             - 2.0 * kappa_S * nu)
                  + (10760.0 - 41600.0 * nu) * chiA2
                  + 21520.0 * delta * chiAS
                  + 40.0 * (269.0 - 36.0 * nu) * chiS2
                  + e4 * (1144.0 * (delta * kappa_A + kappa_S
                                    - 2.0 * kappa_S * nu)
                          + (-963.0 + 4032.0 * nu) * chiA2
                          - 1926.0 * delta * chiAS
                          - 9.0 * (107.0 + 20.0 * nu) * chiS2)
                  + e2 * (14512.0 * (delta * kappa_A + kappa_S
                                     - 2.0 * kappa_S * nu)
                          + 4.0 * (-881.0 + 4064.0 * nu) * chiA2
                          - 7048.0 * delta * chiAS
                          - 4.0 * (881.0 + 540.0 * nu) * chiS2))
    ss_2pn = x**2 * ss_2pn_num / (2432.0 * oome2**2)

    # 2.5PN SO
    so_25pn_num = (
        -64.0 * delta * (57681.0 + 952868.0 * nu) * chi_A
        + 64.0 * (-57681.0 - 985610.0 * nu + 670516.0 * nu2) * chi_S
        + e6 * (3.0 * delta * (3441339.0 + 769244.0 * nu) * chi_A
                + (10324017.0 + 3203016.0 * nu - 817656.0 * nu2) * chi_S)
        + e2 * (-24.0 * delta * (4476734.0 + 1745023.0 * nu) * chi_A
                + 24.0 * (-4476734.0 + 3800571.0 * nu
                          + 3015082.0 * nu2) * chi_S)
        + e4 * (4.0 * delta * (7388082.0 + 4790765.0 * nu) * chi_A
                + (29552328.0 + 67302476.0 * nu
                   + 9741032.0 * nu2) * chi_S)
        + sqrt_oome2 * (
            -8171520.0 * delta * (-3.0 + nu) * chi_A
            + 4085760.0 * (6.0 - 8.0 * nu + nu2) * chi_S
            + e2 * (4919040.0 * delta * (-3.0 + nu) * chi_A
                    - 2459520.0 * (6.0 - 8.0 * nu + nu2) * chi_S)
            + e4 * (3252480.0 * delta * (-3.0 + nu) * chi_A
                    - 1626240.0 * (6.0 - 8.0 * nu + nu2) * chi_S)
        )
    )
    so_25pn = -x**2.5 * so_25pn_num / (1225728.0 * oome2**2.5)

    # 3PN SS
    ss_3pn_num = (
        -192.0 * (delta * kappa_A * (-88517.0 + 333088.0 * nu)
                  + kappa_S * (-88517.0 + 510122.0 * nu
                               - 426748.0 * nu2))
        + 32.0 * (2069461.0 - 9635809.0 * nu + 4395216.0 * nu2) * chiA2
        + 64.0 * delta * (2069461.0 - 3985373.0 * nu) * chiAS
        + 32.0 * (2069461.0 - 6612781.0 * nu + 2014516.0 * nu2) * chiS2
        + e4 * (
            -24.0 * (delta * kappa_A * (2157205.0 + 1979068.0 * nu)
                     + kappa_S * (2157205.0 - 2335342.0 * nu
                                  - 2790004.0 * nu2))
            + 12.0 * (3131923.0 - 13837483.0 * nu
                      + 1693552.0 * nu2) * chiA2
            + 24.0 * delta * (3131923.0 - 2182831.0 * nu) * chiAS
            + 12.0 * (3131923.0 - 3055871.0 * nu
                      + 1466108.0 * nu2) * chiS2
        )
        + e6 * (
            6.0 * (delta * kappa_A * (271948.0 - 604513.0 * nu)
                   + kappa_S * (271948.0 - 1148409.0 * nu
                                + 873600.0 * nu2))
            + (19192791.0 - 80303280.0 * nu
               + 9886464.0 * nu2) * chiA2
            + 6.0 * delta * (6397597.0 - 4003818.0 * nu) * chiAS
            + 3.0 * (6397597.0 - 6830264.0 * nu
                     + 1593536.0 * nu2) * chiS2
        )
        + e2 * (
            -192.0 * (25.0 * delta * kappa_A * (20864.0 + 26341.0 * nu)
                      + kappa_S * (521600.0 - 384675.0 * nu
                                   - 947247.0 * nu2))
            + 16.0 * (-9106018.0 + 34227913.0 * nu
                      + 5762232.0 * nu2) * chiA2
            - 32.0 * delta * (9106018.0 + 3284449.0 * nu) * chiAS
            + 16.0 * (-9106018.0 - 4372739.0 * nu
                      + 3135692.0 * nu2) * chiS2
        )
        + sqrt_oome2 * (
            -192.0 * (delta * kappa_A * (-88517.0 + 333088.0 * nu)
                      + kappa_S * (-88517.0 + 510122.0 * nu
                                   - 426748.0 * nu2))
            + 32.0 * (2069461.0 - 9635809.0 * nu
                      + 4395216.0 * nu2) * chiA2
            + 64.0 * delta * (2069461.0 - 3985373.0 * nu) * chiAS
            + 32.0 * (2069461.0 - 6612781.0 * nu
                      + 2014516.0 * nu2) * chiS2
            + e6 * (
                -18.0 * (delta * kappa_A * (225564.0 + 88571.0 * nu)
                         + kappa_S * (225564.0 - 362557.0 * nu
                                      - 155680.0 * nu2))
                + 27.0 * (379573.0 - 1588880.0 * nu
                          + 185472.0 * nu2) * chiA2
                + 18.0 * delta * (1138719.0 - 521486.0 * nu) * chiAS
                + 9.0 * (1138719.0 - 831208.0 * nu
                         + 169792.0 * nu2) * chiS2
            )
            + e4 * (
                -24.0 * (delta * kappa_A * (2515885.0 + 1850968.0 * nu)
                         + kappa_S * (2515885.0 - 3180802.0 * nu
                                      - 2636284.0 * nu2))
                + 12.0 * (2004643.0 - 9123403.0 * nu
                          + 1078672.0 * nu2) * chiA2
                + 24.0 * delta * (2004643.0 - 1260511.0 * nu) * chiAS
                + 12.0 * (2004643.0 - 1416191.0 * nu
                          + 1056188.0 * nu2) * chiS2
            )
            + e2 * (
                -576.0 * (45.0 * delta * kappa_A * (3312.0 + 5075.0 * nu)
                          + kappa_S * (149040.0 - 69705.0 * nu
                                       - 326389.0 * nu2))
                + 16.0 * (-7701538.0 + 28354633.0 * nu
                          + 6528312.0 * nu2) * chiA2
                - 32.0 * delta * (7701538.0 + 4433569.0 * nu) * chiAS
                + 16.0 * (-7701538.0 - 6415619.0 * nu
                          + 3646412.0 * nu2) * chiS2
            )
        )
    )
    ss_3pn = (x**3 * ss_3pn_num
              / (1225728.0 * oome2**3 * (1.0 + sqrt_oome2)))

    # 3PN non-spin
    log_arg = 2.0 * oome2 * sqrt_x / (1.0 + sqrt_oome2)
    log_term = math.log(log_arg)

    c0 = 64.0 * (-641828882523.0
                  + 109482468480.0 * EULER_GAMMA
                  + 380870611275.0 * nu
                  + 30985883775.0 * nu2
                  + 12481353500.0 * nu3
                  - 727650.0 * PI**2 * (49216.0 + 17917.0 * nu)
                  + 102648712320.0 * LOG2
                  + 116761130640.0 * LOG3)

    c2 = 32.0 * (-6068102350278.0
                  + 792146234880.0 * EULER_GAMMA
                  - 1938077318100.0 * nu
                  + 1564141592475.0 * nu2
                  + 186429656875.0 * nu3
                  + 5821200.0 * PI**2 * (-44512.0 + 8077.0 * nu)
                  + 7937977259520.0 * LOG2
                  - 2860647700680.0 * LOG3)

    c4 = 12.0 * (-17189583356238.0
                  + 1017565274880.0 * EULER_GAMMA
                  + 6546104892250.0 * nu
                  + 5184425896650.0 * nu2
                  + 597064314000.0 * nu3
                  + 121275.0 * PI**2 * (-2744576.0 + 790193.0 * nu)
                  - 224809728380160.0 * LOG2
                  + 42510781647180.0 * LOG3
                  + 72413085937500.0 * LOG5)

    c6 = 4.0 * (4502397710583.0
                + 122366946240.0 * EULER_GAMMA
                + 11968410346200.0 * nu
                + 1488261297600.0 * nu2
                + 392650720000.0 * nu3
                + 363825.0 * PI**2 * (-110016.0 + 9061.0 * nu)
                - 584404081430400.0 * LOG2
                + 159670846150200.0 * LOG3
                + 144826171875000.0 * LOG5)

    c8 = 175.0 * e8 * (46932564429.0
                        - 4051745280.0 * nu
                        - 2937623040.0 * nu2
                        + 178984960.0 * nu3)

    s0 = 64.0 * (-637780831131.0
                  + 109482468480.0 * EULER_GAMMA
                  + 380870611275.0 * nu
                  + 30985883775.0 * nu2
                  + 12481353500.0 * nu3
                  - 727650.0 * PI**2 * (49216.0 + 17917.0 * nu)
                  + 102648712320.0 * LOG2
                  + 116761130640.0 * LOG3)

    s2 = 160.0 * (-1094927702262.0
                   + 158429246976.0 * EULER_GAMMA
                   - 239784537300.0 * nu
                   + 275391017055.0 * nu2
                   + 37285931375.0 * nu3
                   + 291060.0 * PI**2 * (-178048.0 + 28413.0 * nu)
                   + 1587595451904.0 * LOG2
                   - 572129540136.0 * LOG3)

    s4 = 12.0 * (-13196290007886.0
                  + 1017565274880.0 * EULER_GAMMA
                  + 5406255720250.0 * nu
                  + 5078720666250.0 * nu2
                  + 597064314000.0 * nu3
                  + 121275.0 * PI**2 * (-2744576.0 + 865223.0 * nu)
                  - 224809728380160.0 * LOG2
                  + 42510781647180.0 * LOG3
                  + 72413085937500.0 * LOG5)

    s6 = 4.0 * (566190348111.0
                + 122366946240.0 * EULER_GAMMA
                + 9153699093000.0 * nu
                + 3117359044800.0 * nu2
                + 392650720000.0 * nu3
                + 3274425.0 * PI**2 * (-12224.0 + 6519.0 * nu)
                - 584404081430400.0 * LOG2
                + 159670846150200.0 * LOG3
                + 144826171875000.0 * LOG5)

    s8 = 175.0 * e8 * (10539419949.0
                        + 3285893952.0 * nu
                        + 1302605568.0 * nu2
                        + 178984960.0 * nu3)

    sqrt_block = sqrt_oome2 * (s0 + s2 * e2 + s4 * e4 + s6 * e6 + s8)

    log_poly = (24608.0 + 89024.0 * e2 + 42884.0 * e4 + 1719.0 * e6)
    log_block = 284739840.0 * log_poly * (1.0 + sqrt_oome2) * log_term

    pn3_num = c8 + c0 + c2 * e2 + c4 * e4 + c6 * e6 + sqrt_block + log_block

    term_3pn = (-x**3 * pn3_num
                / (169885900800.0 * oome2**3 * (1.0 + sqrt_oome2)))

    bracket = (term_0pn + tail_15pn + so_15pn + term_1pn
               + tail_25pn + so_25pn_tail + term_2pn + ss_2pn
               + so_25pn + ss_3pn + term_3pn)

    return prefactor * bracket


@njit(cache=True, fastmath=True)
def _xdot_resum_numba(e, x, nu, chi_S, chi_A, delta, kappa_S, kappa_A):
    e2 = e * e
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e4 * e4
    e10 = e8 * e2
    nu2 = nu * nu
    nu3 = nu2 * nu
    oome2 = 1.0 - e2
    sqrt_oome2 = math.sqrt(oome2)
    sqrt_x = math.sqrt(x)

    chiA2 = chi_A * chi_A
    chiS2 = chi_S * chi_S
    chiAS = chi_A * chi_S

    prefactor = 2.0 * x**5 * nu / 3.0

    # 0PN
    term_0pn = (96.0 + 292.0 * e2 + 37.0 * e4) / (5.0 * oome2**3.5)

    # 1.5PN tail
    tail_15pn = (PI * x**1.5
                 * (36864.0 + 264000.0 * e2 + 188880.0 * e4 + 10007.0 * e6)
                 / (480.0 * oome2**5))

    # 1.5PN SO
    so_15pn_chiA = ((-5424.0 - 12536.0 * e2 + 2602.0 * e4
                     + 747.0 * e6) * delta * chi_A)
    so_15pn_chiS = ((-9.0 * e6 * (-83.0 + 8.0 * nu)
                     + 48.0 * (-113.0 + 76.0 * nu)
                     + 8.0 * e2 * (-1567.0 + 1670.0 * nu)
                     + e4 * (2602.0 + 4072.0 * nu)) * chi_S)
    so_15pn = x**1.5 * (so_15pn_chiA + so_15pn_chiS) / (30.0 * oome2**5)

    # 1PN
    pn1_num = (16.0 * (743.0 + 924.0 * nu)
               + e6 * (6931.0 + 2072.0 * nu)
               + 14.0 * e4 * (7079.0 + 3690.0 * nu)
               + 8.0 * e2 * (15411.0 + 11158.0 * nu))
    term_1pn = -x * pn1_num / (280.0 * oome2**4.5)

    # 2.5PN tail
    tail_25pn_num = (7.0 * e8 * (-151281.0 + 10007.0 * nu)
                     + 576.0 * (4159.0 + 15876.0 * nu)
                     + 576.0 * e2 * (115991.0 + 171104.0 * nu)
                     + 4.0 * e6 * (17257369.0 + 7471473.0 * nu)
                     + 9.0 * e4 * (18599341.0 + 14964748.0 * nu))
    tail_25pn = -PI * x**2.5 * tail_25pn_num / (20160.0 * oome2**6)

    # 2.5PN SO tail
    so_25pn_tail_chiA = (-4.0 * (129600.0 + 704832.0 * e2
                                  - 29742.0 * e4 - 339871.0 * e6
                                  + 147.0 * e8) * delta * chi_A)
    so_25pn_tail_chiS = ((294.0 * e8 * (-2.0 + nu)
                          + 2304.0 * (-225.0 + 148.0 * nu)
                          + 192.0 * e2 * (-14684.0 + 13789.0 * nu)
                          + 24.0 * e4 * (4957.0 + 101614.0 * nu)
                          + e6 * (1359484.0 + 214925.0 * nu)) * chi_S)
    so_25pn_tail = (PI * x**3
                    * (so_25pn_tail_chiA + so_25pn_tail_chiS)
                    / (720.0 * oome2**6.5))

    # 2PN
    pn2_num = (6048.0 * sqrt_oome2
               * (-48.0 - 250.0 * e2 + 219.0 * e4 + 79.0 * e6)
               * (-5.0 + 2.0 * nu)
               + 3.0 * e8 * (734703.0 + 290664.0 * nu + 58016.0 * nu2)
               + 32.0 * (-11257.0 + 141093.0 * nu + 59472.0 * nu2)
               + 6.0 * e6 * (5905155.0 + 6204657.0 * nu + 1292312.0 * nu2)
               + 16.0 * e2 * (-2678686.0 + 5601690.0 * nu
                              + 1331295.0 * nu2)
               + 12.0 * e4 * (896914.0 + 11637378.0 * nu
                              + 2585233.0 * nu2))
    term_2pn = x**2 * pn2_num / (30240.0 * oome2**5.5)

    # 2PN SS
    ss_2pn_num = (
        8.0 * (480.0 + 2064.0 * e2 + 1064.0 * e4 + 33.0 * e6)
        * (delta * kappa_A + kappa_S - 2.0 * kappa_S * nu)
        + (48.0 * (81.0 - 320.0 * nu)
           + 3.0 * e6 * (-199.0 + 832.0 * nu)
           - 8.0 * e2 * (-865.0 + 3232.0 * nu)
           + 2.0 * e4 * (-1969.0 + 8704.0 * nu)) * chiA2
        + 2.0 * (3888.0 + 6920.0 * e2 - 3938.0 * e4
                 - 597.0 * e6) * delta * chiAS
        + (48.0 * (81.0 - 4.0 * nu)
           - 3.0 * e6 * (199.0 + 36.0 * nu)
           - 8.0 * e2 * (-865.0 + 228.0 * nu)
           - 2.0 * e4 * (1969.0 + 828.0 * nu)) * chiS2
    )
    ss_2pn = x**2 * ss_2pn_num / (40.0 * oome2**5.5)

    # 2.5PN SO
    so_25pn_num = (
        delta * (4008832.0 - 6230784.0 * nu
                 - 224.0 * e4 * (80924.0 + 13833.0 * nu)
                 + 9.0 * e8 * (125467.0 + 27916.0 * nu)
                 - 32.0 * e2 * (642396.0 + 696941.0 * nu)
                 + e6 * (9793328.0 + 4249140.0 * nu)) * chi_A
        + (-9.0 * e8 * (-125467.0 - 41528.0 * nu + 8008.0 * nu2)
           + 128.0 * (31319.0 - 91900.0 * nu + 26544.0 * nu2)
           + 448.0 * e4 * (-40462.0 + 79817.0 * nu + 36774.0 * nu2)
           + 4.0 * e6 * (2448332.0 + 3133943.0 * nu + 337722.0 * nu2)
           + 32.0 * e2 * (-642396.0 - 125759.0 * nu
                          + 632758.0 * nu2)) * chi_S
        + oome2**1.5 * (
            -3584.0 * e2 * (304.0 + 121.0 * e2) * delta
            * (-3.0 + nu) * chi_A
            + 1792.0 * e2 * (304.0 + 121.0 * e2)
            * (6.0 - 8.0 * nu + nu2) * chi_S
        )
    )
    so_25pn = -x**2.5 * so_25pn_num / (6720.0 * oome2**6)

    # 3PN SS
    ss_3pn_kappa = (
        6.0 * delta * kappa_A * (
            192.0 * (-8963.0 + 13706.0 * nu)
            + 3.0 * e8 * (41268.0 + 24115.0 * nu)
            + 96.0 * e2 * (55623.0 + 164080.0 * nu)
            + 4.0 * e6 * (1175865.0 + 952294.0 * nu)
            + 8.0 * e4 * (1721169.0 + 2248456.0 * nu)
        )
        - 6.0 * kappa_S * (
            192.0 * (8963.0 - 31632.0 * nu + 14784.0 * nu2)
            + 3.0 * e8 * (-41268.0 + 58421.0 * nu + 44576.0 * nu2)
            + 96.0 * e2 * (-55623.0 - 52834.0 * nu + 223944.0 * nu2)
            + 8.0 * e4 * (-1721169.0 + 1193882.0 * nu
                          + 3223136.0 * nu2)
            + e6 * (-4703460.0 + 5597744.0 * nu + 5745488.0 * nu2)
        )
    )
    ss_3pn_chiA2 = (
        (-576.0 * (55817.0 - 243029.0 * nu + 59136.0 * nu2)
         - 9.0 * e8 * (503031.0 - 2069104.0 * nu + 112000.0 * nu2)
         - 12.0 * e6 * (2460015.0 - 10429817.0 * nu + 589232.0 * nu2)
         - 32.0 * e2 * (-593801.0 + 851591.0 * nu + 4835712.0 * nu2)
         - 8.0 * e4 * (-8736761.0 + 31182647.0 * nu
                       + 8270976.0 * nu2))
        * chiA2
    )
    ss_3pn_chiAS = (
        2.0 * delta * (
            576.0 * (-55817.0 + 68481.0 * nu)
            + 9.0 * e8 * (-503031.0 + 141694.0 * nu)
            + 12.0 * e6 * (-2460015.0 + 897617.0 * nu)
            + 32.0 * e2 * (593801.0 + 3657185.0 * nu)
            + 8.0 * e4 * (8736761.0 + 6220865.0 * nu)
        ) * chiAS
    )
    ss_3pn_chiS2 = (
        (-576.0 * (55817.0 - 117201.0 * nu + 33460.0 * nu2)
         - 9.0 * e8 * (503031.0 - 226408.0 * nu + 57792.0 * nu2)
         - 12.0 * e6 * (2460015.0 - 1205477.0 * nu + 797748.0 * nu2)
         - 32.0 * e2 * (-593801.0 - 5790757.0 * nu + 1987972.0 * nu2)
         - 8.0 * e4 * (-8736761.0 - 8677333.0 * nu
                       + 5503540.0 * nu2))
        * chiS2
    )
    poly_48 = 48.0 - 358.0 * e2 + 147.0 * e4 + 163.0 * e6
    ss_3pn_sqrt = sqrt_oome2 * (
        -1344.0 * poly_48 * (delta * kappa_A * (-14.0 + 5.0 * nu)
                             + kappa_S * (-14.0 + 33.0 * nu
                                          - 6.0 * nu2))
        + 2688.0 * poly_48 * (11.0 - 46.0 * nu + 6.0 * nu2) * chiA2
        - 5376.0 * poly_48 * delta * (-11.0 + 9.0 * nu) * chiAS
        + 2688.0 * poly_48 * (11.0 - 16.0 * nu + 4.0 * nu2) * chiS2
    )

    ss_3pn_total = (ss_3pn_kappa + ss_3pn_chiA2 + ss_3pn_chiAS
                    + ss_3pn_chiS2 + ss_3pn_sqrt)
    ss_3pn = -x**3 * ss_3pn_total / (20160.0 * oome2**6.5)

    # 3PN non-spin
    log_arg = 2.0 * oome2 * sqrt_x / (1.0 + sqrt_oome2)
    log_term = math.log(log_arg)

    c10 = 175.0 * e10 * (3116030391.0 + 1005041664.0 * nu
                          + 405402624.0 * nu2 + 58347520.0 * nu3)

    sq_inner = (16.0 * (19954466.0 + 75.0 * (-19748.0 + 861.0 * PI**2) * nu
                        - 1990800.0 * nu2)
                + e2 * (2474694912.0 + (964412000.0
                        - 7705950.0 * PI**2) * nu
                        - 341678400.0 * nu2)
                + 3.0 * e4 * (652068196.0
                              + 175.0 * (-960472.0 + 6027.0 * PI**2) * nu
                              + 39620000.0 * nu2)
                + e6 * (-558162656.0
                        + 25.0 * (-18306272.0 + 140343.0 * PI**2) * nu
                        + 239618400.0 * nu2)
                + 300.0 * e8 * (-428445.0 + 70634.0 * nu
                                + 50176.0 * nu2))
    sqrt_3pn = 11088.0 * sqrt_oome2 * sq_inner

    d0 = 640.0 * (-15399771333.0
                   + 1366751232.0 * EULER_GAMMA
                   + 22047056185.0 * nu
                   + 501236505.0 * nu2
                   + 181265700.0 * nu3
                   - 436590.0 * PI**2 * (1024.0 + 1845.0 * nu)
                   + 2733502464.0 * LOG2)

    d2 = 32.0 * (-3181561351866.0
                  + 387246182400.0 * EULER_GAMMA
                  - 9693202750.0 * nu
                  + 636423999750.0 * nu2
                  + 75206939500.0 * nu3
                  - 1819125.0 * PI**2 * (69632.0 + 9717.0 * nu)
                  + 72893399040.0 * LOG2
                  + 700566783840.0 * LOG3)

    d4 = 8.0 * (-28536072962442.0
                 + 2944779425280.0 * EULER_GAMMA
                 + 1156154672750.0 * nu
                 + 7217960245650.0 * nu2
                 + 913931903500.0 * nu3
                 + 363825.0 * PI**2 * (-2647552.0 + 591261.0 * nu)
                 + 65437201589760.0 * LOG2
                 - 31525505272800.0 * LOG3)

    d6 = 12.0 * (664772613120.0 * EULER_GAMMA
                  + 121275.0 * PI**2 * (-1793024.0 + 783633.0 * nu)
                  + 2.0 * (-3371321595729.0
                           + 2173303676775.0 * nu
                           + 1510891888275.0 * nu2
                           + 207243883000.0 * nu3
                           - 220951471910400.0 * LOG2
                           + 53938777308570.0 * LOG3
                           + 60344238281250.0 * LOG5))

    d8 = 14.0 * (358019945973.0
                  + 18121656960.0 * EULER_GAMMA
                  + 906482016000.0 * nu
                  + 351807667200.0 * nu2
                  + 50935808000.0 * nu3
                  + 92619450.0 * PI**2 * (-64.0 + 41.0 * nu)
                  + 36243313920.0 * LOG2)

    log_poly = (3072.0 + 43520.0 * e2 + 82736.0 * e4
                + 28016.0 * e6 + 891.0 * e8)
    log_block = 284739840.0 * log_poly * log_term

    pn3_num = (c10 + sqrt_3pn + d0 + d2 * e2 + d4 * e4 + d6 * e6
               + d8 * e8 + log_block)

    term_3pn = -x**3 * pn3_num / (2794176000.0 * oome2**6.5)

    bracket = (term_0pn + tail_15pn + so_15pn + term_1pn
               + tail_25pn + so_25pn_tail + term_2pn + ss_2pn
               + so_25pn + ss_3pn + term_3pn)

    return prefactor * bracket


def edot_resum(e, x, nu, chi_S=0.0, chi_A=0.0, delta=None,
               kappa_S=0.0, kappa_A=0.0):
    """Public wrapper for resummed de/dt."""
    if delta is None:
        import math as _m
        arg = 1.0 - 4.0 * nu
        delta = _m.sqrt(max(arg, 0.0))
    return _edot_resum_numba(e, x, nu, chi_S, chi_A, delta, kappa_S, kappa_A)


def xdot_resum(e, x, nu, chi_S=0.0, chi_A=0.0, delta=None,
               kappa_S=0.0, kappa_A=0.0):
    """Public wrapper for resummed dx/dt."""
    if delta is None:
        import math as _m
        arg = 1.0 - 4.0 * nu
        delta = _m.sqrt(max(arg, 0.0))
    return _xdot_resum_numba(e, x, nu, chi_S, chi_A, delta, kappa_S, kappa_A)


@njit(cache=True, fastmath=True)
def rhs_resum_inner(e_val, x_val, zeta_val, nu, chi_S, chi_A,
                    delta, kappa_S, kappa_A):
    """Pure numba RHS returning (de, dx, dz) as a tuple."""
    if e_val < 0.0 or x_val <= 0.0 or x_val > 1.0 or e_val >= 1.0:
        return (0.0, 0.0, 0.0)
    de = _edot_resum_numba(e_val, x_val, nu, chi_S, chi_A, delta,
                           kappa_S, kappa_A)
    dx = _xdot_resum_numba(e_val, x_val, nu, chi_S, chi_A, delta,
                           kappa_S, kappa_A)
    dz = _zetadot_numba(e_val, x_val, zeta_val, nu, chi_S, chi_A,
                        delta, kappa_S, kappa_A)
    return (de, dx, dz)


def _rhs(t, y, nu, chi_S, chi_A, delta, kappa_S, kappa_A):
    """ODE RHS compatible with solve_ivp."""
    de, dx, dz = rhs_resum_inner(y[0], y[1], y[2], nu, chi_S, chi_A,
                                 delta, kappa_S, kappa_A)
    return [de, dx, dz]


def integrate_eob_eccentric_dynamics_resum(q, chi1=0.0, chi2=0.0, e0=0.1,
                                           zeta0=0.0, t_eval=None,
                                           omega0=0.0085, t_end=None,
                                           rtol=1e-10, atol=1e-12,
                                           kappa1=1.0, kappa2=1.0,
                                           x_stop=None):
    """Integrate eccentric dynamics using resummed 3PN fluxes (numba)."""
    return integrate_dynamics(
        _rhs, rhs_resum_inner, q, chi1, chi2, e0, zeta0, t_eval,
        omega0, t_end, rtol, atol, kappa1, kappa2, x_stop)


def warmup():
    """Trigger JIT compilation."""
    rhs_resum_inner(0.1, 0.01, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0)
