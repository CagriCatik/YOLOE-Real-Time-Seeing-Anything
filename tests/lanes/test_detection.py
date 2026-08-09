

# --------------------------------------------------------------------------- #
# Rueckfall auf die aeusserste Linie je Seite                                  #
# --------------------------------------------------------------------------- #
def _line(x_bottom: float, role: str = "unknown"):
    """Eine Linie, die am unteren Bildrand an dieser Stelle steht."""
    from adascope.lanes.detection import LaneLine
    L = LaneLine(m=0.0, b=x_bottom, x_bottom=x_bottom, support=10)
    L.role = role
    return L


def test_a_single_line_left_of_ego_still_yields_a_pair():
    """Der Defekt, der vier Aufnahmen totgelegt hat.

    `classify_lanes` vergibt `left_solid` nur an Linien WEITER AUSSEN als die
    ego-naechste. Faehrt das Ego links aussen, ist die Fahrbahnkante zugleich
    die ego-naechste Linie -- sie heisst `left_dashed`, und `left_solid`
    existiert im ganzen Video nicht.

    Gemessen auf `adjusting_speed_scenario_9`: 0 % Homographie bei durchgehend
    sichtbaren Randlinien. Nach dem Rueckfall 100 %.
    """
    from adascope.lanes.bev import outer_solid_pair
    from adascope.lanes.detection import LaneResult

    kante = _line(100.0, "left_dashed")          # einzige Linie links vom Ego
    rechts_innen = _line(700.0, "right_dashed")
    rechts_aussen = _line(1100.0, "right_solid")
    result = LaneResult(lines=[kante, rechts_innen, rechts_aussen],
                        ego_left=kante, ego_right=rechts_innen)

    pair = outer_solid_pair(result)
    assert pair is not None
    assert pair[0] is kante and pair[1] is rechts_aussen


def test_an_existing_solid_role_keeps_precedence_over_the_fallback():
    """Material, das heute laeuft, muss exakt gleich weiterlaufen."""
    from adascope.lanes.bev import outer_solid_pair
    from adascope.lanes.detection import LaneResult

    aussen = _line(50.0, "left_solid")
    innen = _line(300.0, "left_dashed")
    r_innen = _line(700.0, "right_dashed")
    r_aussen = _line(1100.0, "right_solid")
    result = LaneResult(lines=[aussen, innen, r_innen, r_aussen],
                        ego_left=innen, ego_right=r_innen)

    pair = outer_solid_pair(result)
    assert pair[0] is aussen and pair[1] is r_aussen


def test_two_nearly_identical_lines_are_refused_instead_of_warping():
    """Eine entartete Homographie verzerrt still, statt zu scheitern."""
    from adascope.lanes.bev import outer_solid_pair
    from adascope.lanes.detection import LaneResult

    a, b = _line(500.0, "left_dashed"), _line(520.0, "right_dashed")
    result = LaneResult(lines=[a, b], ego_left=a, ego_right=b)
    assert outer_solid_pair(result) is None


def test_no_line_on_one_side_still_yields_nothing():
    """Der Rueckfall darf keine Seite erfinden, die es nicht gibt."""
    from adascope.lanes.bev import outer_solid_pair
    from adascope.lanes.detection import LaneResult

    rechts = _line(700.0, "right_dashed")
    result = LaneResult(lines=[rechts], ego_left=None, ego_right=rechts)
    assert outer_solid_pair(result) is None
