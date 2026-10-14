# STAT 841 / CM 763 Paper Registration, Fall 2026

Registration service: https://stat841-fall2026-paper-registration.aghodsib.chatgpt.site/

Instructor dashboard: https://stat841-fall2026-paper-registration.aghodsib.chatgpt.site/admin

A representative registers one to four members, including the representative.
Member names use `Last name, First name`, for example `Chen, Alex`; the service
checks this format for new registrations and roster edits.
The system assigns Group 1, Group 2, and so on automatically. There is no
user-entered group name or class code. Group numbers are never reused or
renumbered after deletion. Registration
does not require a paper, and the group is counted immediately. The group saves a
private management link and later reserves one listed paper or proposes another.
Listed papers are reserved immediately. Outside proposals require instructor
approval. Rejecting a proposal retains the group and returns feedback.

The dashboard exports all groups, sets presentation times, closes group
registration independently from paper selection, corrects members, releases
papers, replaces lost links, and deletes mistaken groups. Do not put private
management links or student records in this repository.
The service does not send emails. Names and Waterloo email addresses are
self-reported, not university single sign-on.

The course-page section is enabled only when the page has paper_registration:
stat841. It contains one short introduction and a link to the registration
service. The service is the single source of papers and live availability.

The compact student interface uses Raleway and the main course site's blue,
white, and gray palette. Registration is at the top; the 20-paper list is a
closed disclosure below. Individual paper descriptions expand on demand.
There are no search fields, side panels, or internally scrolling lists.
The /register and /group URLs remain compatible with saved links.

The service has a separate Sites source repository and D1 database. Group tokens
appear in private URL fragments; only hashes are stored. Member details require
the group's token or instructor access. Duplicate normalized email addresses
are rejected; different aliases for the same person require instructor review.

For STAT 940, use an independent service/database, credentials, course labels,
paper list, group-size policy, and topic guidance. Keep the same two-stage
workflow and outside-paper approval process.
