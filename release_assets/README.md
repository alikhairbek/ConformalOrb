# Release assets (not tracked by git)

`ConformalOrb_F1_sample.db` (14 MB, SQLite) holds the one-step-corrected Fock matrices F₁ = F[P̂] of the 200 molecules
of the timing experiment (100 per checkpoint), packed as the upper triangle in float32,  Attach it to the GitHub Release of the tagged version (or to the Zenodo deposit)
rather than committing it; the `.gitignore` excludes this folder.

Schema: table `f1(id INTEGER, ckpt TEXT, Ham BLOB, PRIMARY KEY (id, ckpt))` — `Ham` is the packed float32 upper triangle of F₁ in the PySCF def2-SVP AO order; unpack with `src/co_phase3.py:_unpack(np.frombuffer(blob, np.float32), n)` where `n` is the number of basis functions (`src/co_phase3.py:_n_orb(Z)`).
