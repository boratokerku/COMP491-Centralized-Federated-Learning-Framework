import pytest
import torch

from cfl.core.strategy import ClientUpdate, FedAvg, FedProx, build_strategy, proximal_term


def _update(cid: str, value: float, n: int) -> ClientUpdate:
    return ClientUpdate(cid, {"w": torch.full((2,), value), "b": torch.tensor([value])}, n)


def test_fedavg_weighted_by_num_samples():
    updates = [_update("a", 1.0, 10), _update("b", 4.0, 30)]
    result = FedAvg().aggregate({}, updates)
    # (1*10 + 4*30) / 40 = 3.25
    assert torch.allclose(result["w"], torch.full((2,), 3.25))
    assert torch.allclose(result["b"], torch.tensor([3.25]))


def test_fedavg_rejects_empty_and_mismatched():
    with pytest.raises(ValueError):
        FedAvg().aggregate({}, [])
    bad = ClientUpdate("c", {"other": torch.zeros(2)}, 5)
    with pytest.raises(ValueError):
        FedAvg().aggregate({}, [_update("a", 1.0, 5), bad])


def test_fedprox_client_config_and_proximal_term():
    strategy = build_strategy("fedprox", mu=0.5)
    assert isinstance(strategy, FedProx)
    assert strategy.client_config() == {"proximal_mu": 0.5}

    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(3.0)
    term = proximal_term(model, {"weight": torch.tensor([[1.0]])}, mu=0.5)
    # 0.5 * 0.5 * (3 - 1)^2 = 1.0
    assert torch.isclose(term, torch.tensor(1.0))


def test_unknown_strategy():
    with pytest.raises(ValueError):
        build_strategy("fedsgd")
