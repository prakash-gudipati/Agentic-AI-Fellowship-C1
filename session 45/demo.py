"""
Session 45 demo — run the language-rich model and watch the invariants fire.
No API key, no framework, no network. Pure domain model.

    python demo.py
"""
from domain_model import (
    Subscriber, MealPlan, DeliveryWindow, Meal, Money, Order, OrderError,
)


def line(title):
    print("\n" + "=" * 60); print(title); print("=" * 60)


def main():
    line("1 · Build the model in the business's own words")
    asha = Subscriber("sub-001", "Asha", MealPlan(meals_per_week=3))
    window = DeliveryWindow("Tue", "18:00", "20:00")
    order = Order("ord-4821", asha, window)
    print(f"Subscriber : {asha.name}  (plan: {asha.plan.meals_per_week} meals/week)")
    print(f"Window     : {window}")
    print(f"Order      : {order.order_id}  state={order.state.value}")

    line("2 · Invariant I1 — cannot confirm an empty Order")
    try:
        order.confirm()
    except OrderError as e:
        print(f"[blocked correctly] {e}")

    line("3 · Add meals, then confirm")
    order.add_meal(Meal("Paneer bowl", Money(22000)))   # ₹220.00
    order.add_meal(Meal("Rice bowl",   Money(18000)))   # ₹180.00
    order.confirm()
    print(f"Confirmed. {len(order.meals)} meals, total {order.total()}, state={order.state.value}")

    line("4 · Invariant I2 — cannot place before confirm (try out of order)")
    order2 = Order("ord-4822", asha, window)
    order2.add_meal(Meal("Dal bowl", Money(16000)))
    try:
        order2.place()                 # skipped confirm
    except OrderError as e:
        print(f"[blocked correctly] {e}")
    order2.confirm(); order2.place()
    print(f"order2 placed in the right order -> state={order2.state.value}")

    line("5 · Same data, anemic model (the AI's default)")
    from anemic_antipattern import UserManager
    um = UserManager()
    um.process_data("sub-001", ["Paneer bowl", "Rice bowl"], {"window": "Tue 6-8"})
    um.update("sub-001", "placed")     # placed without ever confirming — no rule stops it
    print("UserManager.data:", um.data["sub-001"])
    print(">> 'placed' set directly. No invariant. No language. Works — and is wrong.")

    print("\nDone. The rich model speaks MealHop. The anemic one speaks nothing.")


if __name__ == "__main__":
    main()
