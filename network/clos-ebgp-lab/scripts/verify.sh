#!/usr/bin/env bash
#
# verify.sh - end-to-end sanity check for the Clos eBGP lab
#
# Run after `clab deploy -t clos.clab.yml` has fully come up.
# Allow ~30s for BGP convergence before running.
#
set -u

NODES_SPINE=(spine1 spine2)
NODES_LEAF=(leaf1 leaf2 leaf3 leaf4)
NODES_HOST=(host1 host2 host3 host4)

# Containerlab prefixes the container names with "clab-<labname>-"
LAB_PREFIX="clab-clos-ebgp-"

c() {
  # run a vtysh command inside an FRR node
  local node=$1; shift
  docker exec "${LAB_PREFIX}${node}" vtysh -c "$*"
}

h() {
  # run a shell command inside a host container
  local node=$1; shift
  docker exec "${LAB_PREFIX}${node}" "$@"
}

hr() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

hr "1. BGP neighbor summary on each spine"
for n in "${NODES_SPINE[@]}"; do
  printf '\n--- %s ---\n' "$n"
  c "$n" "show bgp ipv4 unicast summary"
done

hr "2. BGP neighbor summary on each leaf"
for n in "${NODES_LEAF[@]}"; do
  printf '\n--- %s ---\n' "$n"
  c "$n" "show bgp ipv4 unicast summary"
done

hr "3. Full BGP table on leaf1 (should see all leaf loopbacks + host nets)"
c leaf1 "show bgp ipv4 unicast"

hr "4. ECMP check: routes to leaf2's loopback from leaf1"
echo "Expect 2 next-hops (one via spine1, one via spine2):"
c leaf1 "show ip route 10.0.0.12/32"

hr "5. ECMP check: route to host3 net (10.13.0.0/24) from leaf1"
c leaf1 "show ip route 10.13.0.0/24"

hr "6. AS_PATH verification"
echo "Path from leaf1 to leaf3's loopback should be (65000|65001) 65013:"
c leaf1 "show bgp ipv4 unicast 10.0.0.13/32"

hr "7. End-to-end ping: host1 -> host3"
h host1 ping -c 3 -W 2 10.13.0.10 || true

hr "8. End-to-end ping: host2 -> host4"
h host2 ping -c 3 -W 2 10.14.0.10 || true

hr "9. ECMP in action: traceroute host1 -> host3 (same flow may hash to the same spine)"
h host1 traceroute -n -q 1 10.13.0.10 || true
h host1 traceroute -n -q 1 10.13.0.10 || true

hr "10. Convergence timing: ip route count on each leaf"
for n in "${NODES_LEAF[@]}"; do
  count=$(c "$n" "show ip route bgp" | grep -c '^B>')
  printf '%s: %s BGP routes installed\n' "$n" "$count"
done

hr "DONE"
