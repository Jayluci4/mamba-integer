"""
Tests for MDTR: Meta-Discovered Ternary RL.

Tests the meta-network architecture, state extraction, action application,
novel prediction targets, and the full meta-training loop.

Run: python -m pytest tests/test_mdtr.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


# =============================================================================
# MINIMAL MOCK MODEL (ternary-compatible)
# =============================================================================

class TinyTernaryModel(nn.Module):
    """Tiny model that mimics ternary weight structure for testing."""

    def __init__(self, vocab_size=50, d_model=16, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model),
            )
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = x + block(x)
        return self.head(x)


# =============================================================================
# TEST IMPORTS
# =============================================================================

class TestImports:
    def test_mdtr_imports(self):
        from src.meta.mdtr import (
            MDTRConfig,
            MDTRState,
            MDTRAction,
            MDTRMetaNetwork,
            TernaryStateExtractor,
            MDTRModulatedTrainer,
            MDTRMetaTrainer,
        )

    def test_package_imports(self):
        from src.meta import (
            MDTRConfig,
            MDTRMetaNetwork,
            TernaryStateExtractor,
        )


# =============================================================================
# TEST CONFIG
# =============================================================================

class TestMDTRConfig:
    def test_defaults(self):
        from src.meta.mdtr import MDTRConfig

        c = MDTRConfig()
        assert c.n_layers == 24
        assert c.n_domains == 5
        assert c.lstm_hidden == 128
        assert c.lstm_layers == 2
        assert c.n_novel_targets == 16
        assert c.n_parallel_copies == 50

    def test_state_dim(self):
        from src.meta.mdtr import MDTRConfig

        c = MDTRConfig(n_layers=8, n_domains=3, n_boundary_bins=5)
        # 8*5 + 8 + 8 + 3 + 2 = 40 + 8 + 8 + 3 + 2 = 61
        assert c.state_dim == 61

    def test_action_dim(self):
        from src.meta.mdtr import MDTRConfig

        c = MDTRConfig(n_layers=8, n_domains=3, n_novel_targets=4)
        # 8 + 8 + 3 + 8 + 4 = 31
        assert c.action_dim == 31


# =============================================================================
# TEST META-NETWORK
# =============================================================================

class TestMDTRMetaNetwork:
    @pytest.fixture
    def config(self):
        from src.meta.mdtr import MDTRConfig
        return MDTRConfig(
            n_layers=4,
            n_domains=3,
            n_boundary_bins=5,
            lstm_hidden=32,
            n_novel_targets=4,
        )

    @pytest.fixture
    def net(self, config):
        from src.meta.mdtr import MDTRMetaNetwork
        return MDTRMetaNetwork(config)

    def test_architecture(self, net, config):
        assert net.config is config
        assert isinstance(net.lstm, nn.LSTM)
        assert net.lstm.hidden_size == 32
        assert net.lstm.num_layers == 2

    def test_param_count(self, net):
        count = net.get_param_count()
        assert count > 0
        # Small meta-network — should be well under 1M params
        assert count < 100_000

    def test_forward_pass(self, net, config):
        from src.meta.mdtr import MDTRAction

        state = torch.randn(config.state_dim)
        action, novel_preds, hidden = net(state)

        # Action structure
        assert isinstance(action, MDTRAction)
        assert action.lr_multipliers.shape == (config.n_layers,)
        assert action.boundary_widths.shape == (config.n_layers,)
        assert action.domain_weights.shape == (config.n_domains,)
        assert action.flip_penalties.shape == (config.n_layers,)
        assert action.novel_targets.shape == (config.n_novel_targets,)

        # Novel predictions
        assert novel_preds.shape == (config.n_novel_targets,)

        # Hidden state
        assert len(hidden) == 2  # (h, c)
        assert hidden[0].shape == (config.lstm_layers, 1, config.lstm_hidden)

    def test_sequential_steps(self, net, config):
        """LSTM hidden state should carry across steps."""
        states = [torch.randn(config.state_dim) for _ in range(5)]
        hidden = net.init_hidden()

        actions = []
        for s in states:
            action, _, hidden = net(s, hidden)
            actions.append(action.lr_multipliers.clone())

        # Actions should differ across steps (LSTM accumulates context)
        assert not torch.allclose(actions[0], actions[-1])

    def test_action_constraints(self, net, config):
        """Actions should be within valid ranges."""
        state = torch.randn(config.state_dim)
        action, _, _ = net(state)

        # LR multipliers: positive
        assert (action.lr_multipliers > 0).all()
        assert (action.lr_multipliers <= config.lr_multiplier_range[1]).all()

        # Boundary widths: within range
        lo, hi = config.epsilon_range
        assert (action.boundary_widths >= lo - 1e-6).all()
        assert (action.boundary_widths <= hi + 1e-6).all()

        # Domain weights: sum to 1
        assert action.domain_weights.sum().item() == pytest.approx(1.0, abs=1e-5)

        # Flip penalties: non-negative
        assert (action.flip_penalties >= 0).all()

    def test_batched_forward(self, net, config):
        """Should handle batched input."""
        state = torch.randn(3, config.state_dim)
        hidden = net.init_hidden(batch_size=3)
        # Batched forward works through the input projection
        x = torch.relu(net.input_proj(state))
        x = x.unsqueeze(1)
        out, hidden = net.lstm(x, hidden)
        assert out.shape == (3, 1, config.lstm_hidden)


# =============================================================================
# TEST STATE EXTRACTION
# =============================================================================

class TestTernaryStateExtractor:
    @pytest.fixture
    def config(self):
        from src.meta.mdtr import MDTRConfig
        return MDTRConfig(n_layers=4, n_domains=3, n_boundary_bins=5)

    @pytest.fixture
    def extractor(self, config):
        from src.meta.mdtr import TernaryStateExtractor
        return TernaryStateExtractor(config)

    @pytest.fixture
    def model(self):
        return TinyTernaryModel(vocab_size=50, d_model=16, n_layers=4)

    def test_extract_state(self, extractor, model, config):
        from src.meta.mdtr import MDTRState

        state = extractor.extract_state(model, total_steps=100)
        assert isinstance(state, MDTRState)
        assert state.boundary_histograms.shape == (4, 5)
        assert state.flip_rates.shape == (4,)
        assert state.grad_norms.shape == (4,)
        assert state.domain_scores.shape == (3,)

    def test_state_to_vector(self, extractor, model, config):
        state = extractor.extract_state(model, total_steps=100)
        vec = state.to_vector()
        assert vec.shape == (config.state_dim,)
        assert not torch.isnan(vec).any()

    def test_boundary_histograms_sum_to_one(self, extractor, model):
        state = extractor.extract_state(model, total_steps=100)
        for i in range(state.boundary_histograms.shape[0]):
            hist = state.boundary_histograms[i]
            if hist.sum() > 0:
                assert hist.sum().item() == pytest.approx(1.0, abs=0.05)

    def test_flip_rates_first_step_zero(self, extractor, model):
        """First step should have zero flip rates (no previous values)."""
        state = extractor.extract_state(model, total_steps=100)
        assert (state.flip_rates == 0).all()

    def test_flip_rates_after_weight_change(self, extractor, model):
        """Flip rates should be > 0 after weights change."""
        extractor.extract_state(model, total_steps=100)

        # Modify weights significantly
        with torch.no_grad():
            for p in model.parameters():
                if p.dim() >= 2:
                    p.add_(torch.randn_like(p) * 5.0)

        state2 = extractor.extract_state(model, total_steps=100)
        # At least some layers should show flips
        assert state2.flip_rates.sum() > 0

    def test_grad_norms_after_backward(self, extractor, model):
        """Grad norms should be nonzero after a backward pass."""
        x = torch.randint(0, 50, (2, 8))
        loss = model(x).sum()
        loss.backward()

        state = extractor.extract_state(model, total_steps=100)
        assert state.grad_norms.sum() > 0

    def test_domain_scores_forwarded(self, extractor, model):
        state = extractor.extract_state(
            model, domain_scores=[0.8, 0.5, 0.3], total_steps=100
        )
        assert state.domain_scores[0].item() == pytest.approx(0.8)
        assert state.domain_scores[1].item() == pytest.approx(0.5)
        assert state.domain_scores[2].item() == pytest.approx(0.3)

    def test_loss_delta(self, extractor, model):
        state = extractor.extract_state(
            model, current_loss=2.0, prev_loss=3.0, total_steps=100
        )
        assert state.loss_delta == pytest.approx(-1.0)

    def test_more_layers_than_config(self):
        """Model with more weight layers than config.n_layers should aggregate."""
        from src.meta.mdtr import TernaryStateExtractor, MDTRConfig

        config = MDTRConfig(n_layers=2, n_domains=3, n_boundary_bins=5)
        extractor = TernaryStateExtractor(config)
        # 8-layer model but n_layers=2
        model = TinyTernaryModel(n_layers=8)
        state = extractor.extract_state(model, total_steps=100)
        assert state.boundary_histograms.shape == (2, 5)


# =============================================================================
# TEST MODULATED TRAINER
# =============================================================================

class TestMDTRModulatedTrainer:
    @pytest.fixture
    def config(self):
        from src.meta.mdtr import MDTRConfig
        return MDTRConfig(n_layers=4, n_domains=3, n_boundary_bins=5)

    @pytest.fixture
    def modulator(self, config):
        from src.meta.mdtr import MDTRModulatedTrainer
        return MDTRModulatedTrainer(config)

    def test_flip_penalty_computation(self, modulator, config):
        from src.meta.mdtr import MDTRAction

        model = TinyTernaryModel(n_layers=4)
        action = MDTRAction(
            lr_multipliers=torch.ones(4),
            boundary_widths=torch.full((4,), 0.3),
            domain_weights=torch.ones(3) / 3,
            flip_penalties=torch.ones(4) * 0.5,
            novel_targets=torch.zeros(config.n_novel_targets),
        )

        penalty = modulator.compute_flip_penalty(model, action)
        assert penalty.item() >= 0.0

    def test_domain_loss_reweighting(self, modulator, config):
        from src.meta.mdtr import MDTRAction

        action = MDTRAction(
            lr_multipliers=torch.ones(4),
            boundary_widths=torch.full((4,), 0.1),
            domain_weights=torch.tensor([0.5, 0.3, 0.2]),
            flip_penalties=torch.zeros(4),
            novel_targets=torch.zeros(config.n_novel_targets),
        )

        domain_losses = {
            "math": torch.tensor(1.0),
            "sql": torch.tensor(2.0),
            "english": torch.tensor(0.5),
        }
        total = modulator.reweight_domain_loss(
            domain_losses, action, ["math", "sql", "english"]
        )
        expected = 0.5 * 1.0 + 0.3 * 2.0 + 0.2 * 0.5
        assert total.item() == pytest.approx(expected, abs=1e-5)

    def test_modulated_step(self, modulator, config):
        from src.meta.mdtr import MDTRAction

        model = TinyTernaryModel(n_layers=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        action = MDTRAction(
            lr_multipliers=torch.ones(4),
            boundary_widths=torch.full((4,), 0.1),
            domain_weights=torch.ones(3) / 3,
            flip_penalties=torch.full((4,), 0.01),
            novel_targets=torch.zeros(config.n_novel_targets),
        )

        # Forward + backward via modulated step
        x = torch.randint(0, 50, (2, 8))
        loss = model(x).sum()
        total = modulator.modulated_step(model, loss, optimizer, action)
        # Total loss is original loss + flip penalty; just verify it ran
        assert isinstance(total, torch.Tensor)
        assert not torch.isnan(total)


# =============================================================================
# TEST NOVEL PREDICTION LOSS
# =============================================================================

class TestNovelPredictions:
    def test_novel_prediction_loss(self):
        from src.meta.mdtr import (
            compute_novel_prediction_loss,
            MDTRConfig,
            MDTRState,
        )

        config = MDTRConfig(n_layers=4, n_domains=3, n_boundary_bins=5, n_novel_targets=4)

        predictions = [torch.randn(4) for _ in range(3)]
        states = [
            MDTRState(
                boundary_histograms=torch.randn(4, 5),
                flip_rates=torch.randn(4),
                grad_norms=torch.randn(4),
                domain_scores=torch.randn(3),
                progress=i / 3.0,
                loss_delta=-0.1,
            )
            for i in range(3)
        ]

        loss = compute_novel_prediction_loss(predictions, states, config)
        assert loss.item() >= 0.0

    def test_empty_predictions(self):
        from src.meta.mdtr import compute_novel_prediction_loss, MDTRConfig

        config = MDTRConfig(n_novel_targets=4)
        loss = compute_novel_prediction_loss([], [], config)
        assert loss.item() == 0.0


# =============================================================================
# TEST META-TRAINER (simplified inner loop)
# =============================================================================

class TestMDTRMetaTrainer:
    @pytest.fixture
    def config(self):
        from src.meta.mdtr import MDTRConfig
        return MDTRConfig(
            n_layers=4,
            n_domains=3,
            n_boundary_bins=5,
            lstm_hidden=16,
            n_novel_targets=4,
            n_inner_steps=10,
            n_meta_steps=2,
            n_parallel_copies=2,
            meta_lr=1e-3,
        )

    @pytest.fixture
    def meta_trainer(self, config):
        from src.meta.mdtr import MDTRMetaNetwork, MDTRMetaTrainer

        meta_net = MDTRMetaNetwork(config)

        def create_model():
            return TinyTernaryModel(vocab_size=50, d_model=16, n_layers=4)

        def evaluate(model):
            # Simple mock evaluation
            with torch.no_grad():
                x = torch.randint(0, 50, (4, 8))
                logits = model(x)
                entropy = -(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            return {"math": 0.5, "sql": 0.3, "english": 0.4}

        return MDTRMetaTrainer(
            meta_network=meta_net,
            config=config,
            create_model_fn=create_model,
            evaluate_fn=evaluate,
            device="cpu",
        )

    def test_inner_loop(self, meta_trainer, config):
        """Inner loop should produce states, actions, and a score."""
        model = TinyTernaryModel(vocab_size=50, d_model=16, n_layers=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        step_count = [0]

        def train_step(m, step):
            x = torch.randint(0, 50, (2, 8))
            logits = m(x)
            loss = F.cross_entropy(
                logits.view(-1, 50), x.view(-1)
            )
            step_count[0] += 1
            return loss, [0.5, 0.3, 0.4]

        states, actions, novel_preds, score = meta_trainer.run_inner_loop(
            train_step_fn=train_step,
            model=model,
            optimizer=optimizer,
            n_steps=5,
        )

        assert len(states) == 5
        assert len(actions) == 5
        assert len(novel_preds) == 5
        assert score > 0  # evaluate_fn returns nonzero

    def test_meta_step(self, meta_trainer):
        """One meta-step should produce metrics."""
        def train_step(m, step):
            x = torch.randint(0, 50, (2, 8))
            logits = m(x)
            loss = F.cross_entropy(logits.view(-1, 50), x.view(-1))
            return loss, [0.5, 0.3, 0.4]

        metrics = meta_trainer.meta_step_fn(train_step)
        assert "meta_step" in metrics
        assert "avg_score" in metrics
        assert "meta_loss" in metrics
        assert metrics["meta_step"] == 1

    def test_full_meta_train(self, meta_trainer):
        """Full meta-training loop should run to completion."""
        def train_step(m, step):
            x = torch.randint(0, 50, (2, 8))
            logits = m(x)
            loss = F.cross_entropy(logits.view(-1, 50), x.view(-1))
            return loss, [0.5, 0.3, 0.4]

        history = meta_trainer.train(train_step, n_meta_steps=2)
        assert len(history) == 2
        assert all("avg_score" in h for h in history)

    def test_checkpoint_save_load(self, meta_trainer, tmp_path):
        """Checkpoint save/load should preserve state."""
        path = str(tmp_path / "mdtr_ckpt.pt")
        meta_trainer.meta_step = 42
        meta_trainer.save_checkpoint(path)

        # Load into new trainer
        from src.meta.mdtr import MDTRMetaNetwork, MDTRMetaTrainer, MDTRConfig
        config = meta_trainer.config
        new_net = MDTRMetaNetwork(config)
        new_trainer = MDTRMetaTrainer(new_net, config)
        new_trainer.load_checkpoint(path)

        assert new_trainer.meta_step == 42


# =============================================================================
# TEST LAYER PARAM GROUPS
# =============================================================================

class TestLayerParamGroups:
    def test_build_groups(self):
        from src.meta.mdtr import build_layer_param_groups

        model = TinyTernaryModel(n_layers=4)
        groups = build_layer_param_groups(model, base_lr=1e-3, n_layers=4)

        # Should have groups for layers + potentially other params
        assert len(groups) > 0
        # Each group has layer_idx
        for g in groups:
            assert "layer_idx" in g

    def test_lr_assignment(self):
        from src.meta.mdtr import build_layer_param_groups

        model = TinyTernaryModel(n_layers=4)
        groups = build_layer_param_groups(model, base_lr=1e-3, n_layers=4)

        for g in groups:
            assert g["lr"] == 1e-3


# =============================================================================
# TEST ACTION PARSING
# =============================================================================

class TestMDTRAction:
    def test_from_vector(self):
        from src.meta.mdtr import MDTRAction, MDTRConfig

        config = MDTRConfig(n_layers=4, n_domains=3, n_novel_targets=4)
        vec = torch.randn(config.action_dim)
        action = MDTRAction.from_vector(vec, config)

        assert action.lr_multipliers.shape == (4,)
        assert action.boundary_widths.shape == (4,)
        assert action.domain_weights.shape == (3,)
        assert action.flip_penalties.shape == (4,)
        assert action.novel_targets.shape == (4,)

    def test_lr_multipliers_positive(self):
        from src.meta.mdtr import MDTRAction, MDTRConfig

        config = MDTRConfig(n_layers=4, n_domains=3)
        # Even with very negative inputs, softplus ensures positivity
        vec = torch.full((config.action_dim,), -10.0)
        action = MDTRAction.from_vector(vec, config)
        assert (action.lr_multipliers > 0).all()

    def test_domain_weights_sum_to_one(self):
        from src.meta.mdtr import MDTRAction, MDTRConfig

        config = MDTRConfig(n_layers=4, n_domains=5)
        for _ in range(10):
            vec = torch.randn(config.action_dim)
            action = MDTRAction.from_vector(vec, config)
            assert action.domain_weights.sum().item() == pytest.approx(1.0, abs=1e-5)


# =============================================================================
# INTEGRATION: END-TO-END MINI PIPELINE
# =============================================================================

class TestE2EIntegration:
    def test_state_to_action_pipeline(self):
        """Full pipeline: model → state → meta-network → action → modulated step."""
        from src.meta.mdtr import (
            MDTRConfig,
            MDTRMetaNetwork,
            TernaryStateExtractor,
            MDTRModulatedTrainer,
        )

        config = MDTRConfig(
            n_layers=4, n_domains=3, n_boundary_bins=5,
            lstm_hidden=16, n_novel_targets=4,
        )

        model = TinyTernaryModel(n_layers=4)
        meta_net = MDTRMetaNetwork(config)
        extractor = TernaryStateExtractor(config)
        modulator = MDTRModulatedTrainer(config)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        hidden = meta_net.init_hidden()

        # Run 3 meta-modulated training steps
        prev_loss = 0.0
        for step in range(3):
            # Forward pass
            x = torch.randint(0, 50, (2, 8))
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, 50), x.view(-1))

            # Extract state
            state = extractor.extract_state(
                model, domain_scores=[0.5, 0.3, 0.4],
                current_loss=loss.item(), prev_loss=prev_loss,
                total_steps=10,
            )

            # Meta-network decision
            state_vec = state.to_vector()
            action, novel_preds, hidden = meta_net(state_vec, hidden)

            # Modulated training step
            total = modulator.modulated_step(model, loss, optimizer, action)

            prev_loss = loss.item()

        # Model should still produce valid outputs
        with torch.no_grad():
            out = model(torch.randint(0, 50, (1, 5)))
        assert out.shape == (1, 5, 50)
        assert not torch.isnan(out).any()


if __name__ == "__main__":
    import torch.nn.functional as F
    pytest.main([__file__, "-v", "--tb=short"])
