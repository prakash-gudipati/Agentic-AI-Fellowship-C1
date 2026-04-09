"""
Demonstrates:
- Lists as ordered containers (store multiple contacts)
- Dictionaries as structured records (each contact has named fields)
- Nested data: a list inside a dictionary (tags)
- Sets for instant deduplication (unique tags across all contacts)
- Tuples as read-only coordinates / records
- Named constants at the top of the file (production pattern)
- .get() for safe dictionary access without crashing

"""

# PRODUCTION PATTERN: Named constants at the top of the file.
# If this value needs changing, you change it in one place — not throughout the code.
MAX_CONTACTS = 100


# This is our "database": a list of dictionaries.
# Each dictionary is one contact record.
# The list holds all the records — ordered, indexable, appendable.
contacts = [
    {
        "name": "Priya Sharma",
        "phone": "1111111111",
        "email": "priya@example.com",
        "tags": ["college", "design"] 
    },
    {
        "name": "Alex Johnson",
        "phone": "1111111111",
        "email": "alex@example.com",
        "tags": ["work", "engineering"]
    },
    {
        "name": "Meera Nair",
        "phone": "1111111111",
        "email": "meera@example.com",
        "tags": ["college", "engineering"]
    }
]

def add_contact(contacts_list,name, phone, email, tags=None):
    """Add a new contact dictionary to the contacts list."""
    # named constant — never hardcode the limit inside the logic
    if len(contacts_list) >= MAX_CONTACTS:
        print(f"Contacts book is full. Maximum allowed contacts is {MAX_CONTACTS}")
        return
    
    # Build the new contact as a dictionary — same structure as all others
    new_contact = {
        "name": name,
        "phone": phone,
        "email": email,
        "tags": tags if tags else [] # default to empty list if no tags given
    }

    contacts_list.append(new_contact)
    print(f"[ADDED] {name} saved to contacts")

def search_contant(contacts_list, search_name):
    """Search for contacts whose name contains the search term. Returns a list."""
    # List comprehension with a condition — reads like English:
    # "give me every contact where the search term appears in their name"
    results = [contact for contact in contacts_list 
               if search_name.lower() in contact['name'].lower()]
    return results

def list_all_contacts(contacts_list):
    # if(len(contacts_list) == 0 or contacts_list == None)
    # Above line works the same way as below line
    if not contacts_list:
        print("[EMPTY] Contacts list is empty")
    
    # sorted() with a key function — Python knows to sort by the "name" field
    sorted_contacts = sorted(contacts_list, key=lambda c : c['name'])
    
    print(f"\n All contacts ({len(sorted_contacts)})")
    for contact in sorted_contacts:
        # .get() returns a default value if the key doesn't exist — safer than contact["phone"]
        print(f" {contact['name']}  ")
        print(f"email: {contact['email']}")
        print(f"phone: {contact['phone']}")
        print(f"tags:  {','.join(contact.get('tags', []))}")


def get_all_tags(contacts_list):
    """Return a set of every unique tag used across all contacts."""
    # A set automatically removes duplicates — no manual dedup needed
    all_tags = set()
    for contact in contacts_list:
        for tag in contact.get('tags', []):
            all_tags.add(tag)
    return all_tags

def delete_contact(contacts_list, name_to_delete):
    """Delete the first contact whose name matches exactly."""
    for i, contact in enumerate(contacts_list):
        if contact['name'].lower() == name_to_delete.lower():
            removed = contacts_list.pop(i)
            print(f"[DELETED] Removed {name_to_delete} from contacts list")
            return
    print(f"{name_to_delete} is not found in the contacts list")

def main():
    print(" ======  Contacnt Book ======")

    # Show starting state
    list_all_contacts(contacts)

    # Add a new contact
    print("\n Adding a new contact")
    add_contact(contacts, "Prakash", "1111111111", "prakash@example.com", ['office', 'yoga'])

    # Search for a contact
    print("\n Searching for Prakash")
    results = search_contant(contacts, "prakash")
    for result in results:
        print(f"Found {result['name']} - {result['email']}")
    
    # Show all unique tags — set removes duplicates automatically
    print("\n Get all the unique tags")
    tags = get_all_tags(contacts)
    print(f"All the tags that are in use: {sorted(tags)}")

    # Delete a contact
    print("\n Deleting a contact")
    delete_contact(contacts, "prakash")

    # Final state
    print("\n Final list of contancts")
    list_all_contacts(contacts)


if __name__ == "__main__":
    main()