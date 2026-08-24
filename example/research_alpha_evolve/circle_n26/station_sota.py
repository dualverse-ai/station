#!/usr/bin/env python3
# Copyright 2025 DualverseAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Verification script for the best circle packing configurations.
This script loads the optimal configurations found by the agents and verifies
they pass Google's verification function.
"""

import numpy as np
import itertools

# Best N=32 configuration with score 2.93957277120630689
# Shape: (32, 3), dtype: float64
# Data loaded inline from station_data_circle_n32/rooms/research/internal/packings/Verity_II_2.939573_653.npz
CIRCLE_N32 = np.array([[0.5913254062324885 , 0.9366847467936634 , 0.06331525320633571],
       [0.11156670822435097, 0.11156670822435097, 0.1115667082243506 ],
       [0.9269330385282407 , 0.07306696147175928, 0.0730669614717591 ],
       [0.09657220783995829, 0.6962447847055132 , 0.09657220783995783],
       [0.593163108162112  , 0.41073608188695826, 0.09216464924936667],
       [0.9094306251117177 , 0.5957813052526898 , 0.09056937488828189],
       [0.5287196651622721 , 0.09859039863160775, 0.09859039863160714],
       [0.9302289243170231 , 0.9302289243170231 , 0.06977107568297632],
       [0.2572358559381545 , 0.40852495848126563, 0.09392542960110455],
       [0.2644856421727376 , 0.9376127924467954 , 0.06238720755320436],
       [0.4260881970914672 , 0.691866929252117  , 0.09567929075561363],
       [0.7533825213033992 , 0.3226385997517136 , 0.09067797900211039],
       [0.7515575922102232 , 0.6905962077363336 , 0.09358751542673652],
       [0.42710036218527536, 0.28662900075069314, 0.11515009501531481],
       [0.09380120532914625, 0.3161647698578795 , 0.09380120532914582],
       [0.7427114977388672 , 0.11611806282627356, 0.1161180628262731 ],
       [0.2605631271104417 , 0.5973508891925817 , 0.09492981348853938],
       [0.4256085805916491 , 0.4989810453519822 , 0.09720718943396014],
       [0.5903732992652366 , 0.7817741164904765 , 0.0915983029745469 ],
       [0.6095545318756697 , 0.24805883339705917, 0.07133631694839393],
       [0.3334957393177939 , 0.09664323745292071, 0.0966432374529204 ],
       [0.42730226163744306, 0.893771284957308  , 0.10622871504269184],
       [0.7511110411155143 , 0.505155837326994  , 0.0918533926422035 ],
       [0.7566648166220216 , 0.8920594989217282 , 0.10794050107827137],
       [0.589663242565222  , 0.5965224714665011 , 0.09365470284345875],
       [0.9094712287480968 , 0.41468316366433244, 0.09052877125190299],
       [0.10365414473978905, 0.8963458552602109 , 0.1036541447397887 ],
       [0.09485861738505687, 0.5048216292250877 , 0.09485861738505674],
       [0.9124287442499188 , 0.7738967048279848 , 0.08757125575008091],
       [0.2630302142908603 , 0.7837484224916332 , 0.09148404581147697],
       [0.24372294258660382, 0.24142932918255153, 0.07371569947520341],
       [0.910575349816327  , 0.23473312947973796, 0.08942465018367254]])

# Best N=26 configuration with score 2.63598308491754763
# Shape: (26, 3), dtype: float64
# Data loaded inline from station_data_circle_n26/rooms/research/internal/packings/Quest_II_2.635983_850.npz
CIRCLE_N26 = np.array([[0.08463950069577039, 0.08463950069577512, 0.08463950069577039],
       [0.1302211010652235 , 0.29460948878182674, 0.13022110106522078],
       [0.07886037291596529, 0.4972844462041077 , 0.07886037291596246],
       [0.13325857277081246, 0.7023095250891577 , 0.13325857277081066],
       [0.08492626245489716, 0.9150737375450997 , 0.08492626245489716],
       [0.27478328335082264, 0.10679014462858295, 0.10679014462858152],
       [0.3869235534096072 , 0.29474605904894985, 0.11207708895025231],
       [0.27534261677144267, 0.4955317606743549 , 0.11762968804654121],
       [0.38166584445264595, 0.7026096036990235 , 0.11514888016001965],
       [0.2739528396239525 , 0.8948174397312528 , 0.10518256026874717],
       [0.48460080265192135, 0.1030605201415824 , 0.10306052014157979],
       [0.5976347963876669 , 0.2716298514860001 , 0.099898350592751  ],
       [0.5299634197531942 , 0.49866807550340425, 0.1370104301237466 ],
       [0.5960427019081332 , 0.7269057143115987 , 0.10060036781870768],
       [0.48259558221054105, 0.8965327666420502 , 0.10346723335794983],
       [0.6832585349730863 , 0.09573232930701864, 0.09573232930701764],
       [0.7636735693833866 , 0.23971052792753683, 0.06918067635723386],
       [0.7424170495016363 , 0.40335878360267935, 0.09584232574550254],
       [0.7420494434605915 , 0.5952197329397375 , 0.09601897575825113],
       [0.6820800429325875 , 0.9038486659542355 , 0.09615133404576207],
       [0.8892209872092853 , 0.11077901279071509, 0.11077901279071417],
       [0.9076084484290412 , 0.31311580997040517, 0.09239155157095676],
       [0.9060726627225547 , 0.49942836913904337, 0.09392733727744268],
       [0.9074079050485652 , 0.6859430219867868 , 0.09259209495143315],
       [0.8888438205895587 , 0.8888438205895572 , 0.11115617941043988],
       [0.7629588636540018 , 0.7593524015609378 , 0.06944019371125516]])


def verify_circles(circles: np.ndarray):
    """Checks that the circles are disjoint and lie inside a unit square.

    Args:
      circles: A numpy array of shape (num_circles, 3), where each row is
        of the form (x, y, radius), specifying a circle.

    Raises:
      AssertionError if the circles are not disjoint or do not lie inside the
      unit square.
    """
    # Check pairwise disjointness.
    for circle1, circle2 in itertools.combinations(circles, 2):
        center_distance = np.sqrt((circle1[0] - circle2[0])**2 + (circle1[1] - circle2[1])**2)
        radii_sum = circle1[2] + circle2[2]
        assert center_distance >= radii_sum, f"Circles are NOT disjoint: {circle1} and {circle2}."

    # Check all circles lie inside the unit square [0,1]x[0,1].
    for circle in circles:
        assert (0 <= min(circle[0], circle[1]) - circle[2] and max(circle[0],circle[1]) + circle[2] <= 1), f"Circle {circle} is NOT fully inside the unit square."


if __name__ == "__main__":
    configs = [
        ("N=32", CIRCLE_N32, 2.93957277120630689),
        ("N=26", CIRCLE_N26, 2.63598308491754763)
    ]

    all_passed = True

    for name, config, expected_score in configs:
        print(f"=" * 60)
        print(f"Verifying {name} configuration...")
        print(f"Configuration shape: {config.shape}")
        print(f"Configuration dtype: {config.dtype}")
        print(f"Number of circles: {len(config)}")
        actual_score = np.sum(config[:, 2])
        print(f"Sum of radii: {actual_score:.17f}")
        print(f"Expected score: {expected_score:.17f}")
        print()

        try:
            verify_circles(config)
            print(f"✓ {name} VERIFICATION PASSED!")
            print("  - All circles are disjoint")
            print("  - All circles lie inside the unit square [0,1]x[0,1]")
            print(f"  - Total sum of radii: {actual_score:.17f}")
            print()
        except AssertionError as e:
            print(f"✗ {name} VERIFICATION FAILED!")
            print(f"  Error: {e}")
            print()
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("✓ ALL CONFIGURATIONS PASSED VERIFICATION!")
    else:
        print("✗ SOME CONFIGURATIONS FAILED!")
        exit(1)
