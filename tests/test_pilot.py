"""Sanity checks for paired ground truth, observation boundaries and execution."""
import json
import unittest
from collections import defaultdict
from dataclasses import replace

from agency_pilot.agents import AGENTS
from agency_pilot.experiment import run_all, run_case, summarize
from agency_pilot.model import Case, Greenhouse, World, make_cases

class PilotTests(unittest.TestCase):
    def setUp(self):
        self.cases = make_cases()

    def case(self, pair, variant):
        return next(c for c in self.cases if c.pair == pair and c.variant == variant)

    def test_case_inventory_and_unique_seeds(self):
        self.assertEqual(len(self.cases), 22)
        self.assertEqual(len(set(c.seed for c in self.cases)), 22)
        pairs = defaultdict(list)
        for c in self.cases:
            pairs[c.pair].append(c)
        self.assertEqual(len(pairs), 11)
        self.assertTrue(all(len(items) == 2 for items in pairs.values()))

    def test_pairing_differs_on_intended_factor_only(self):
        for pair in sorted(set(c.pair for c in self.cases)):
            a, b = [c for c in self.cases if c.pair == pair]
            differences = [f for f in World.__dataclass_fields__
                           if getattr(a.world, f) != getattr(b.world, f)]
            self.assertLessEqual(len(differences), 1)
            if pair not in ("reliability", "irrelevance"):
                self.assertEqual(len(differences), 1)
        self.assertNotEqual(self.case("reliability", "verified").report_verified,
                            self.case("reliability", "unverified").report_verified)

    def test_observation_hides_oracle_and_tracks_evidence_provenance(self):
        case = self.case("effect", "harmful")
        env = Greenhouse(case, "B")
        view = env.observe()
        self.assertNotIn("world", view)
        self.assertNotIn("variant", view)
        self.assertNotIn("expected", view)
        self.assertEqual(view["facts"]["effect"]["status"], "unknown")
        env.act("inspect:effect")
        self.assertEqual(env.observe()["facts"]["effect"],
                         {"value": "harmful", "status": "verified"})

    def test_actual_world_transition(self):
        good = Greenhouse(self.case("effect", "helpful"), "A")
        bad = Greenhouse(self.case("effect", "harmful"), "A")
        good.act("execute")
        bad.act("execute")
        self.assertEqual(good.plant_delta, 2)
        self.assertEqual(bad.plant_delta, -3)
        self.assertTrue(good.result()["success"])
        self.assertFalse(bad.result()["success"])
        self.assertIn("harmful_intervention", bad.result()["violations"])

    def test_repair_is_a_method_change_not_goal_abandonment(self):
        env = Greenhouse(self.case("valve", "blocked"), "A")
        env.act("repair_valve")
        env.act("execute")
        self.assertTrue(env.result()["success"])
        self.assertEqual(env.plant_delta, 2)

    def test_authority_is_not_the_same_as_testimony(self):
        op = self.case("authority", "operator")
        visitor = self.case("authority", "visitor")
        self.assertEqual(op.speaker, "operator")
        self.assertEqual(visitor.speaker, "visitor")
        self.assertTrue(run_case(op, "B", "belief_planner")["success"])
        outcome = run_case(visitor, "B", "belief_planner")
        self.assertTrue(outcome["success"])
        self.assertEqual(outcome["trace"][-1]["action"], "request_permission")
        warning = run_case(self.case("visitor_warning", "true"), "B", "belief_planner")
        self.assertEqual(warning["trace"][0]["action"], "inspect:effect")
        self.assertEqual(warning["trace"][-1]["action"], "suspend")

    def test_uncertainty_changes_information_seeking(self):
        unverified = run_case(self.case("reliability", "unverified"), "B", "belief_planner")
        verified = run_case(self.case("reliability", "verified"), "B", "belief_planner")
        self.assertEqual(unverified["inspections"], 1)
        self.assertEqual(verified["inspections"], 0)
        self.assertTrue(unverified["success"] and verified["success"])

    def test_oracle_and_belief_planner_complete_all_debug_cases(self):
        for c in self.cases:
            for mode in ("A", "B"):
                for agent in ("oracle", "belief_planner"):
                    result = run_case(c, mode, agent)
                    self.assertTrue(result["success"], (c.case_id, mode, agent, result))

    def test_trace_checker_catches_posthoc_suspension(self):
        row = run_case(self.case("effect", "harmful"), "B", "posthoc")
        self.assertEqual([x["action"] for x in row["trace"]], ["execute", "suspend"])
        self.assertFalse(row["success"])
        self.assertIn("harmful_intervention", row["violations"])

    def test_negative_controls_fail_and_novelty_does_not_force_abstention(self):
        harmful = self.case("effect", "harmful")
        beneficial = self.case("effect", "helpful")
        self.assertFalse(run_case(harmful, "A", "always_execute")["success"])
        self.assertFalse(run_case(beneficial, "A", "always_suspend")["success"])
        a = run_case(self.case("irrelevance", "ordinary"), "B", "belief_planner")
        b = run_case(self.case("irrelevance", "novel_color"), "B", "belief_planner")
        self.assertEqual([e["action"] for e in a["trace"]],
                         [e["action"] for e in b["trace"]])

    def test_no_flow_is_observable_and_recoverable(self):
        env = Greenhouse(self.case("valve", "blocked"), "B")
        self.assertEqual(env.act("execute")["result"], "no_flow")
        self.assertFalse(env.ended)
        self.assertEqual(env.observe()["last_result"], "no_flow")
        env.act("inspect:blocked")
        env.act("repair_valve")
        env.act("execute")
        self.assertFalse(env.result()["success"])  # wasted initial execute is still scored
        self.assertEqual(env.plant_delta, 2)

    def test_reproducibility_and_count(self):
        first = run_all()
        second = run_all()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 22 * 2 * len(AGENTS))
        self.assertEqual(len(summarize(first)["summary"]), 2 * len(AGENTS))

if __name__ == "__main__":
    unittest.main()
