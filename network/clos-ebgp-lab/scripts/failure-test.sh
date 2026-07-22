#!/usr/bin/env bash
#
# failure-test.sh - exercise failure domain & convergence
#
# Scenarios:
#   1) Bring down leaf1 <-> spine1 link; observe leaf1 still reaches everything via spine2.
#   2) Shut down all spine1 fabric links; verify the fabric reconverges through spine2.
#   3) Restore and confirm symmetric reconvergence.
#
set -u

LAB_PREFIX="clab-clos-ebgp-"

c() { docker exec "${LAB_PREFIX}$1" vtysh -c "$2"; }
h() { docker exec "${LAB_PREFIX}$1" ${@:2}; }
sh() { docker exec "${LAB_PREFIX}$1" sh -c "$2"; }
hr() { printf '\n\033[1;33m==> %s\033[0m\n' "$*"; }

hr "Baseline: leaf1 routes to leaf3 loopback (expect 2-way ECMP)"
c leaf1 "show ip route 10.0.0.13/32"

hr "Start background ping host1 -> host3 (5 pps for 60s)"
docker exec -d "${LAB_PREFIX}host1" sh -c 'ping -i 0.2 -c 300 10.13.0.10 > /tmp/ping.log 2>&1'
sleep 2

hr "Scenario 1: shut leaf1<->spine1 link (leaf1:eth1)"
date +"%T.%3N start"
sh leaf1 "ip link set dev eth1 down"
date +"%T.%3N link down issued"
sleep 5

hr "leaf1 routes to leaf3 (expect single path via spine2)"
c leaf1 "show ip route 10.0.0.13/32"

hr "leaf1 BGP summary (spine1 peering should be Idle/Active)"
c leaf1 "show bgp ipv4 unicast summary"

hr "Restore link"
sh leaf1 "ip link set dev eth1 up"
sleep 8

hr "After restore: leaf1 routes to leaf3 (expect 2-way ECMP again)"
c leaf1 "show ip route 10.0.0.13/32"

hr "Stop ping & summarize loss"
sleep 3
sh host1 "cat /tmp/ping.log | tail -5"
echo
echo "Loss percentage:"
sh host1 "grep 'packet loss' /tmp/ping.log || true"

hr "Scenario 2: bring down all spine1 fabric links"
date +"%T.%3N spine1 fabric links down"
sh spine1 "for i in eth1 eth2 eth3 eth4; do ip link set dev \$i down; done"
sleep 5

hr "leaf1 BGP summary (only spine2 should be Established)"
c leaf1 "show bgp ipv4 unicast summary"

hr "leaf1 routes to leaf3 (single path via spine2)"
c leaf1 "show ip route 10.0.0.13/32"

hr "Ping host1 -> host3 should still succeed"
h host1 ping -c 3 -W 2 10.13.0.10 || true

hr "Restore spine1 fabric links"
sh spine1 "for i in eth1 eth2 eth3 eth4; do ip link set dev \$i up; done"
echo "Waiting 15s for spine1 BGP to come up..."
sleep 15
c leaf1 "show bgp ipv4 unicast summary"
c leaf1 "show ip route 10.0.0.13/32"

hr "DONE"
