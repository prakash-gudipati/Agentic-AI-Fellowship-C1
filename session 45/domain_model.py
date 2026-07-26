"""
Session 45 — Domain-Driven Design (the language-rich model)
Bounded context: ORDERING (MealHop)

This is what Claude Code generates when you feed it GLOSSARY.md + context_map.md.
Every class, method, and field uses the business's own words — the ubiquitous
language. Compare with anemic_antipattern.py to see what the AI writes WITHOUT a
glossary.

Invariants enforced here:
  I1. A confirmed Order must contain at least one Meal.
  I2. An Order cannot be placed until it has been confirmed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


# ── VALUE OBJECTS ────────────────────────────────────────────────────────────
# Value objects have NO identity. Two with the same values ARE the same thing,
# like two ₹10 notes. We make them immutable (frozen) so they can't drift.

@dataclass(frozen=True)
class Money:
    """Value object: an amount in a currency. Equal Moneys are interchangeable."""
    amount_paise: int          # store minor units (paise/cents) — never floats for money
    currency: str = "INR"

    def __post_init__(self):
        if self.amount_paise < 0:
            raise ValueError("Money cannot be negative")

    def __str__(self) -> str:
        symbol = {"INR": "₹", "USD": "$"}.get(self.currency, "")
        return f"{symbol}{self.amount_paise / 100:.2f}"


@dataclass(frozen=True)
class MealPlan:
    """Value object: a subscription tier — N meals per week."""
    meals_per_week: int

    def __post_init__(self):
        if self.meals_per_week not in (3, 5, 7):
            raise ValueError("MealPlan must be 3, 5, or 7 meals/week")


@dataclass(frozen=True)
class DeliveryWindow:
    """Value object: a named time slot a delivery may arrive in."""
    day: str                   # e.g. "Tue"
    start: str                 # e.g. "18:00"
    end: str                   # e.g. "20:00"

    def __str__(self) -> str:
        return f"{self.day} {self.start}–{self.end}"


@dataclass(frozen=True)
class Meal:
    """Value object: one dish chosen for a slot."""
    name: str
    price: Money


# ── ENTITIES ─────────────────────────────────────────────────────────────────
# Entities have identity. The SAME Subscriber even after they move house;
# the SAME Order #4821 even as meals are added.

@dataclass
class Subscriber:
    """Entity: a person who holds a subscription. Identity = subscriber_id."""
    subscriber_id: str
    name: str
    plan: MealPlan


class OrderState(Enum):
    DRAFT = "draft"
    CONFIRMED = "confirmed"
    PLACED = "placed"


@dataclass
class Order:
    """
    Entity + aggregate root for the Ordering context.
    Identity = order_id. Guards the invariants for the whole basket.
    """
    order_id: str
    subscriber: Subscriber
    window: DeliveryWindow
    _meals: list[Meal] = field(default_factory=list)
    state: OrderState = OrderState.DRAFT

    # Domain actions — named for what the business DOES, not for CRUD.
    def add_meal(self, meal: Meal) -> None:
        if self.state is not OrderState.DRAFT:
            raise OrderError("Can only add meals while the Order is a draft")
        self._meals.append(meal)

    def confirm(self) -> None:
        # Invariant I1: a confirmed Order must contain at least one Meal.
        if not self._meals:
            raise OrderError("Cannot confirm an Order with no meals")
        self.state = OrderState.CONFIRMED

    def place(self) -> None:
        # Invariant I2: an Order cannot be placed until it is confirmed.
        if self.state is not OrderState.CONFIRMED:
            raise OrderError("Cannot place an Order before it is confirmed")
        self.state = OrderState.PLACED

    def total(self) -> Money:
        return Money(sum(m.price.amount_paise for m in self._meals))

    @property
    def meals(self) -> tuple[Meal, ...]:
        return tuple(self._meals)   # expose a copy — the basket is guarded


class OrderError(Exception):
    """Raised when a business rule (invariant) would be broken."""
