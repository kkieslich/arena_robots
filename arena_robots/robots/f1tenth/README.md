# f1tenth

F1TENTH 1/10 race car: Ackermann steering, 0.58 x 0.31 m, 3.74 kg. The first car-like robot in Arena, it cannot turn in place.

Geometry vendored from [f1tenth/f1tenth_gym_ros](https://github.com/f1tenth/f1tenth_gym_ros) `launch/ego_racecar.xacro` (MIT, Copyright (c) 2020 Hongrui Zheng). Changes: frames unprefixed, front hinges turned into revolute steering joints, wheel axes on y, body box lengthened to the 0.58 m body of f1tenth_gym, collisions, inertials and joint limits added. Vehicle limits come from the parameters of [f1tenth_gym](https://github.com/f1tenth/f1tenth_gym): bicycle steering angle `s_max` 0.4189 rad, `v_max` 20 m/s, `a_max` 9.51 m/s2, tyre friction `mu` 1.0489.

Driven by `ackermann_steering_controller` from ros2_controllers. `base_link` sits on the rear axle centre, the reference point of the controller. Wheelbase 0.3302 m, track 0.2413 m, wheel radius 0.0508 m. Each steering joint reaches the wheel angles the controller commands at `s_max`: 0.4887 rad as the inner wheel and 0.3657 rad as the outer wheel, so a saturated command keeps the Ackermann geometry and the bicycle limit is the minimum turning radius.

Nav2 defaults to MPPI with Smac Hybrid-A* (`nav2:` block of `caps/mobile.yaml`) instead of the bringup's NavFn with Regulated Pure Pursuit, which rotates in place.

Differences from the [build guide](https://f1tenth.readthedocs.io/en/main/getting_started/build_car/index.html) car on a Traxxas 1/10 4WD chassis:

- The real car is 4WD; this model drives the rear axle only, the fixed axle of `ackermann_steering_controller`.
- Dimensions follow `ego_racecar.xacro` and f1tenth_gym (wheelbase 330 mm).
- Lidar: the `hokuyo_ust20lx` component stands in for the build's Hokuyo UST-10LX, listed as Hokuyo 10LX in the F1TENTH master BOM (same body and 270 deg field of view, 20 m instead of 10 m range).

## License

The vendored geometry is under the MIT license of f1tenth_gym_ros:

```
MIT License

Copyright (c) 2020 Hongrui Zheng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
