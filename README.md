# Generator for basis and matrix files for the ordinary graph complex

All graphs are connected, simple, and have at least trivalent vertices.
The defect of a graph of loop order g is defined such that the number of edges is e=3(g-1)-d, and the number of vertices if v=2(g-1)-d.
In other words, defect zero corresponds to trivalent graphs, and then every edge contraction increases the defect by one.

The program generates 5 kinds of graphs:
 - Plain graphs: These are all simple connected graphs with >= trivalent vertices
 - Odd (edge) graphs: These are all biconnected plain graphs without odd symmetries, in the sign convention of the Kontsevich graph complex GC_2.
 - Even (edge) graphs: These are all biconnected plain graphs without odd symmetries, in the sign convention of the Kontsevich graph complex GC_3.
 - Triconnected versions: Same as odd/even before, but additionally requiring that the graphs are triconnected.

Furthermore, it can create the matrices of the differential in the Kontsevich graph complex, i.e. edge contraction.

# Workflow

You have to follow the follwing workflow:
1) Generate plain graphs. Here the graphs of higher defect are produced from those of lower defect, so the lower defect graphs need to be produced first.
Example to generate plain graphs of loop orders 4,5,6,7,8,9 and defects 0,1,2,3,4,5:

```rust_ggen2 plain 4 9 0 5 -t30```

The optional "-t30" allows the program to use up to 30 threadds on multicore machines.
2) Generate even or odd graphs. They are filtered from the plain graphs, so it is assumed the plain graphs have been created before.
Example:

```rust_ggen2 odd 4 9 0 5 -t30 -3```

The optional "-3" tells the program to generate trivalent graphs.
3) Generate matrix files. Here it is assumed that both the source and the target basis has already been created.
Example:

```rust_ggen2 odd 4 9 0 4 -t30 -3 -M```

# Relation to GH

The functionality above is also available in GH. However, the latter is too slow to generate large files. rust_ggen2 on the other hand is intended to produce large basis and matrix files (several 100M basis elements) on multicore machines with enough RAM.
