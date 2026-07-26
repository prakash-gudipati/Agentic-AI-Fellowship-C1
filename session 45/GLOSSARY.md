# MealHop — Ubiquitous Language Glossary
# (the AI-context artifact — feed this to every coding session)
#
# RULE: code uses these words. If the AI writes "User", "Manager", "Service",
# "Data", or "Process", it has left the language. Point it back here.

| Term | Meaning (the ONE meaning) | Context |
|------|---------------------------|---------|
| Subscriber | A person who holds a subscription with MealHop. Has identity. | Ordering / Billing |
| MealPlan | A subscription tier: a number of meals per week (e.g. 3/week). A value. | Ordering |
| Meal | A single dish a subscriber can choose for a slot. | Ordering / Kitchen |
| DeliveryWindow | A named time slot a delivery may arrive in (e.g. Tue 18:00–20:00). A value. | Ordering / Delivery |
| Order | A basket of meals a subscriber is assembling and confirming for a week. | Ordering |
| PrepTicket | What an Order becomes inside the Kitchen: what to cook, how many, by when. | Kitchen |
| Stop | What an Order becomes inside Delivery: a drop on a rider's route. | Delivery |
| Charge | What an Order becomes inside Billing: a line on an invoice. | Billing |
| Recipient | The person who receives a delivery — may differ from the Subscriber (gifts). | Delivery |
| Payer | The Subscriber seen as the party who pays. Has a payment method + billing cycle. | Billing |
| Pause | A subscriber-requested break; paused weeks are not charged and not cooked. | Ordering / Billing |
| Money | An amount + currency. Interchangeable; equal Moneys are the same. A value. | All |

**Most overloaded term:** "Order" — it means four different things, one per context.
We do not force one definition. We give each context its own, and translate at the
boundary (see context_map.md).
