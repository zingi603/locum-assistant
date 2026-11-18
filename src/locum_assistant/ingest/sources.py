"""The documents the assistant answers from. Public guidance only: English versions of the
GPhC standards and guidance, the Responsible Pharmacist Regulations 2008 and the 2025 Order that
changed the law on supervision.

legislation.gov.uk only holds the 2008 Regulations as originally made, so the 2025 Order is
ingested too and the assistant is told to prefer it where they differ. The GPhC supervision FAQ
page is left out: the site returns 403 to a bot user agent, and we identify ourselves honestly
rather than pose as a browser.

The files are downloaded at ingest time and never committed, because GPhC documents are
GPhC copyright. Legislation is Crown copyright under the Open Government Licence."""

from dataclasses import dataclass
from typing import Literal

GPHC = "https://assets.pharmacyregulation.org/files/"


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    url: str
    kind: Literal["pdf", "html", "legislation"]
    publisher: str = "General Pharmaceutical Council"


SOURCES: list[Source] = [
    Source("gphc-standards-professionals", "Standards for pharmacy professionals",
           GPHC + "standards_for_pharmacy_professionals_may_2017_0.pdf", "pdf"),
    Source("gphc-standards-pharmacies", "Standards for registered pharmacies",
           GPHC + "document/standards_for_registered_pharmacies_june_2018_0.pdf", "pdf"),
    Source("gphc-safe-effective-team", "Guidance to ensure a safe and effective pharmacy team",
           GPHC + "2025-10/gphc-guidance-to-ensure-safe-effective-pharmacy-team-october-2025.pdf", "pdf"),
    Source("gphc-confidentiality", "In practice: guidance on confidentiality",
           GPHC + "2025-12/gphc-in-practice-guidance-on-confidentiality-updated-november-2025.pdf", "pdf"),
    Source("gphc-consent", "In practice: guidance on consent",
           GPHC + "2025-06/gphc-in-practice-guidance-on-consent-updated-may-2025.pdf", "pdf"),
    Source("gphc-raising-concerns", "In practice: guidance on raising concerns",
           GPHC + "2024-01/in-practice-guidance-on-raising-concerns-november-2020.pdf", "pdf"),
    Source("gphc-religion-beliefs", "In practice: guidance on religion, personal values and beliefs",
           GPHC + "2024-01/in_practice-_guidance_on_religion_personal_values_and_beliefs.pdf", "pdf"),
    Source("gphc-professionalism-online", "Demonstrating professionalism online",
           GPHC + "2024-11/Demonstrating-professionalism-online-January-2024.pdf", "pdf"),
    Source("gphc-newly-qualified", "Advice for newly qualified pharmacists",
           GPHC + "2026-07/Advice-for-newly-qualified-pharmacists-June-2026a.pdf", "pdf"),
    Source("gphc-prescribers", "In practice: guidance for pharmacist prescribers",
           GPHC + "2026-10/in-practice-guidance-for-pharmacist-prescribers-updated-april-2025a.pdf",
           "pdf"),
    Source("gphc-scope-of-practice", "Scope of practice statement",
           GPHC + "2025-11/gphc-scope-of-practice-statement-october%202025.pdf", "pdf"),
    Source("gphc-ai-position", "Position statement on the use of artificial intelligence in pharmacy",
           GPHC + "2026-04/GPhC_position_statement_on_the_use_of_Artificial_Intelligence_in_pharmacy_0.pdf",
           "pdf"),
    Source("gphc-learning-from-incidents", "Pharmacy team toolkit: learning from incidents",
           GPHC + "2024-01/gphc-pharmacy-team-toolkit-learning-from-incidents-june-2022.pdf", "pdf"),
    Source("gphc-distance-services", "Guidance for registered pharmacies providing services at a distance",
           GPHC + "2026-01/gphc-guidance-registered-pharmacies-providing-pharmacy-services-distance-february-2025%20FINALa.pdf",
           "pdf"),
    Source("rp-regulations", "The Medicines (Pharmacies) (Responsible Pharmacist) Regulations 2008",
           "https://www.legislation.gov.uk/uksi/2008/2789", "legislation", publisher="UK Government"),
    Source("supervision-order-2025",
           "The Human Medicines (Authorisation by Pharmacists and Supervision by Pharmacy "
           "Technicians) Order 2025",
           "https://www.legislation.gov.uk/uksi/2025/1249", "legislation", publisher="UK Government"),
]

BY_ID = {s.id: s for s in SOURCES}
