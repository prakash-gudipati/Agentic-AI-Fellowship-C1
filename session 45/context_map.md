# MealHop — Bounded Contexts + Context Map

## The four bounded contexts

| Context  | Responsible for | "Order" means here | Doesn't care about |
|----------|-----------------|--------------------|--------------------|
| Ordering | What the subscriber wants this week | a basket of meals being assembled & confirmed | routes, recipes, invoices |
| Kitchen  | Cooking the right food on time | a PrepTicket: what to cook, how many, by when | who the subscriber is, price |
| Delivery | Getting food to the door | a Stop on a rider's route | menu, recipe, price |
| Billing  | Charging correctly | a Charge: a line on an invoice | recipes, routes |

## The "Order" collision (why one definition would be wrong)
- Ordering: Order #4821 = "Asha's 3 meals for next week, confirmed."
- Kitchen:  Order #4821 = "2× paneer bowl, 1× rice bowl, ready by 16:00."
- Delivery: Order #4821 = "Stop 7 on Route B, Tue 18:00–20:00, hand to Asha."
- Billing:  Order #4821 = "₹540 line on Asha's October invoice."

## Context map (handoffs)

   [Ordering] --confirmed Order--> [Kitchen]   (becomes a PrepTicket)
       |                                |
       | confirmed Order                | cooked
       v                                v
   [Billing] <--charge--           [Delivery]  (becomes a Stop)

| From     | To       | What crosses          | Translated into |
|----------|----------|-----------------------|-----------------|
| Ordering | Kitchen  | a confirmed Order     | PrepTicket      |
| Ordering | Delivery | a confirmed Order     | Stop            |
| Ordering | Billing  | a confirmed Order     | Charge          |

Each arrow is a TRANSLATION. The word "Order" does not travel unchanged — it is
re-expressed in the receiving context's language. That is what keeps each context clean.
