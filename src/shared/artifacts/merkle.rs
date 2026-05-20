use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MerkleSibling {
    Left(String),
    Right(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MerkleProof {
    leaf_count: usize,
    siblings: Vec<MerkleSibling>,
}

impl MerkleProof {
    pub fn new(leaf_count: usize, siblings: Vec<MerkleSibling>) -> Result<Self> {
        if leaf_count == 0 {
            bail!("Merkle proof requires at least one leaf");
        }
        Ok(Self {
            leaf_count,
            siblings,
        })
    }

    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }

    pub fn siblings(&self) -> &[MerkleSibling] {
        &self.siblings
    }
}

pub fn merkle_root(domain: &[u8], leaves: &[Vec<u8>]) -> String {
    let leaf_hashes = leaves
        .iter()
        .enumerate()
        .map(|(index, payload)| leaf_hash(domain, index, payload))
        .collect::<Vec<_>>();
    merkle_root_from_leaf_hashes(domain, &leaf_hashes)
}

pub fn merkle_proof(domain: &[u8], leaves: &[Vec<u8>], index: usize) -> Result<MerkleProof> {
    if index >= leaves.len() {
        bail!(
            "Merkle proof index {index} is out of range for {} leaves",
            leaves.len()
        );
    }
    let leaf_hashes = leaves
        .iter()
        .enumerate()
        .map(|(leaf_index, payload)| leaf_hash(domain, leaf_index, payload))
        .collect::<Vec<_>>();
    merkle_proof_from_leaf_hashes(domain, &leaf_hashes, index)
}

pub fn verify_merkle_proof(
    domain: &[u8],
    root: &str,
    index: usize,
    payload: &[u8],
    proof: &MerkleProof,
) -> Result<()> {
    if index >= proof.leaf_count {
        bail!(
            "Merkle proof index {index} is out of range for {} leaves",
            proof.leaf_count
        );
    }

    let mut current = leaf_hash(domain, index, payload);
    let mut current_index = index;
    let mut level_size = proof.leaf_count;
    let mut siblings = proof.siblings.iter();

    while level_size > 1 {
        if current_index % 2 == 1 {
            let Some(MerkleSibling::Left(left)) = siblings.next() else {
                bail!("Merkle proof is missing left sibling");
            };
            current = node_hash(domain, left, &current);
        } else if current_index + 1 < level_size {
            let Some(MerkleSibling::Right(right)) = siblings.next() else {
                bail!("Merkle proof is missing right sibling");
            };
            current = node_hash(domain, &current, right);
        }
        current_index /= 2;
        level_size = level_size.div_ceil(2);
    }

    if siblings.next().is_some() {
        bail!("Merkle proof has unused siblings");
    }
    if current != root {
        bail!("Merkle proof root mismatch: {current} vs {root}");
    }
    Ok(())
}

fn merkle_root_from_leaf_hashes(domain: &[u8], leaf_hashes: &[String]) -> String {
    if leaf_hashes.is_empty() {
        return empty_root(domain);
    }

    let mut level = leaf_hashes.to_vec();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        for pair in level.chunks(2) {
            if let [left, right] = pair {
                next.push(node_hash(domain, left, right));
            } else {
                next.push(pair[0].clone());
            }
        }
        level = next;
    }
    level.remove(0)
}

fn merkle_proof_from_leaf_hashes(
    domain: &[u8],
    leaf_hashes: &[String],
    index: usize,
) -> Result<MerkleProof> {
    if index >= leaf_hashes.len() {
        bail!(
            "Merkle proof index {index} is out of range for {} leaves",
            leaf_hashes.len()
        );
    }

    let mut current_index = index;
    let mut level = leaf_hashes.to_vec();
    let mut siblings = Vec::new();
    while level.len() > 1 {
        if current_index % 2 == 1 {
            siblings.push(MerkleSibling::Left(level[current_index - 1].clone()));
        } else if current_index + 1 < level.len() {
            siblings.push(MerkleSibling::Right(level[current_index + 1].clone()));
        }

        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        for pair in level.chunks(2) {
            if let [left, right] = pair {
                next.push(node_hash(domain, left, right));
            } else {
                next.push(pair[0].clone());
            }
        }
        current_index /= 2;
        level = next;
    }

    MerkleProof::new(leaf_hashes.len(), siblings)
}

fn leaf_hash(domain: &[u8], index: usize, payload: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"raster-merkle-leaf-v1");
    hasher.update((domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    hasher.update((index as u64).to_le_bytes());
    hasher.update((payload.len() as u64).to_le_bytes());
    hasher.update(payload);
    hex_digest(hasher.finalize())
}

fn node_hash(domain: &[u8], left: &str, right: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"raster-merkle-node-v1");
    hasher.update((domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    hasher.update(left.as_bytes());
    hasher.update(right.as_bytes());
    hex_digest(hasher.finalize())
}

fn empty_root(domain: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"raster-merkle-empty-v1");
    hasher.update((domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    hex_digest(hasher.finalize())
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOMAIN: &[u8] = b"raster-artifact-token-ids-merkle-v1";

    #[test]
    fn empty_root_is_stable_for_domain() {
        assert_eq!(
            merkle_root(DOMAIN, &[]),
            "840ab97955371685959783a46790111eed287b7a4b4401a1615683382fd2124e"
        );
    }

    #[test]
    fn single_and_multi_leaf_roots_are_stable() {
        assert_eq!(
            merkle_root(DOMAIN, &[b"a".to_vec()]),
            "bf05bd34070dc43e5b2a95057a8f12858225a16f9d78ea50ae69b4c0c97edee8"
        );
        assert_eq!(
            merkle_root(DOMAIN, &[b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]),
            "5347e20800b62fcb27052d1fc21cd38bc6b0e79a0655663870cc3ca775bfa05f"
        );
    }

    #[test]
    fn proof_verifies_requested_leaf() {
        let leaves = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()];
        let root = merkle_root(DOMAIN, &leaves);
        let proof = merkle_proof(DOMAIN, &leaves, 1).expect("proof");

        verify_merkle_proof(DOMAIN, &root, 1, b"b", &proof).expect("valid proof");
    }

    #[test]
    fn proof_rejects_wrong_payload_index_root_and_extra_siblings() {
        let leaves = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()];
        let root = merkle_root(DOMAIN, &leaves);
        let proof = merkle_proof(DOMAIN, &leaves, 1).expect("proof");

        assert!(verify_merkle_proof(DOMAIN, &root, 1, b"x", &proof).is_err());
        assert!(verify_merkle_proof(DOMAIN, &root, 2, b"b", &proof).is_err());
        assert!(verify_merkle_proof(DOMAIN, "not-the-root", 1, b"b", &proof).is_err());

        let mut malformed = proof.clone();
        malformed
            .siblings
            .push(MerkleSibling::Right("extra".to_string()));
        assert!(verify_merkle_proof(DOMAIN, &root, 1, b"b", &malformed).is_err());
    }
}
