"""
Build the knowledge pack: 7 short HR policy PDFs for a FICTIONAL company, Kestrel Data Labs.

Each document is written page by page, so facts sit on known pages. That lets us check
citations like [Leave Policy#2] against a gold answer key.

Run:  python make_corpus.py      -> writes PDFs into corpus/
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path("corpus")
COMPANY = "Kestrel Data Labs Pvt. Ltd. (fictional company created for this project)"

# --- Corpus content: {file_name: (title, [page1_paragraphs, page2_paragraphs, ...])} ---
DOCS = {
    "01_leave_policy.pdf": ("Leave Policy", [
        ["1. Purpose and scope",
         "This policy explains the types of leave available to employees of Kestrel Data Labs and how leave is "
         "requested, approved and recorded. It applies to all full-time and part-time employees in India. "
         "Interns and contractors are covered by their own engagement letters and are not covered by this policy.",
         "2. Leave year",
         "The leave year runs from 1 January to 31 December. All leave balances are tracked in the HR portal, "
         "PeopleDesk, and employees can see their live balance there at any time.",
         "3. Annual (earned) leave",
         "Full-time employees receive 24 days of annual leave per leave year. Annual leave accrues at 2 days per "
         "completed month of service. Part-time employees receive annual leave on a pro-rata basis according to "
         "their contracted hours. New joiners can use annual leave only after completing their 3-month probation."],
        ["4. Carry forward and encashment",
         "Employees may carry forward a maximum of 10 unused annual leave days into the next leave year. Any "
         "unused balance above 10 days lapses on 31 December and is not paid out, except on resignation, when "
         "up to 15 accrued days are encashed at the basic salary rate.",
         "5. Applying for leave",
         "Planned annual leave must be applied for in PeopleDesk at least 7 calendar days in advance. Leave of "
         "more than 5 consecutive working days needs at least 21 days' notice and approval from both the "
         "reporting manager and the department head. Managers should respond to a request within 3 working days.",
         "6. Sick leave",
         "Employees receive 12 days of paid sick leave per leave year. Sick leave cannot be carried forward. A "
         "medical certificate is required for sick leave of more than 2 consecutive days."],
        ["7. Parental and family leave",
         "Maternity leave: 26 weeks of paid leave for the first two children, in line with the Maternity Benefit "
         "Act. Paternity leave: 10 working days of paid leave, to be taken within 6 months of the child's birth "
         "or adoption. Bereavement leave: 5 working days on the death of an immediate family member.",
         "8. Public holidays",
         "The company observes 12 public holidays each year. The holiday list is published in December for the "
         "following year. Employees who are required to work on a public holiday receive a compensatory day off, "
         "to be used within 60 days.",
         "9. Leave without pay",
         "Leave without pay of up to 30 days may be granted once annual leave is exhausted, with department head "
         "and HR approval."]]),

    "02_remote_work_policy.pdf": ("Remote and Hybrid Work Policy", [
        ["1. Working model",
         "Kestrel Data Labs follows a hybrid working model. Employees are expected to work from their assigned "
         "office at least 3 days per week. Teams agree their common office days with their manager; Tuesday and "
         "Thursday are company-wide collaboration days on which all hybrid employees should be in the office.",
         "2. Core hours",
         "Regardless of location, employees must be available online during core hours of 11:00 to 16:00 India "
         "Standard Time (IST), Monday to Friday. Meetings should be scheduled inside core hours where possible.",
         "3. Fully remote roles",
         "A role can be designated fully remote only with approval from the department head and the Chief People "
         "Officer. Fully remote employees must still attend the office for the quarterly planning week."],
        ["4. Home office support",
         "Hybrid and remote employees receive a one-time home office setup allowance of INR 15,000 for a chair, "
         "desk or monitor, claimable within the first 6 months of joining. Employees also receive an internet "
         "allowance of INR 1,500 per month, paid with salary.",
         "5. Working from another location",
         "Employees may work temporarily from another city in India with manager approval. Working from outside "
         "India is limited to 30 days per calendar year and requires prior written approval from HR and the "
         "Security team, because of tax and data-protection rules.",
         "6. Security when working remotely",
         "Remote work must follow the IT and Information Security Policy, including using the company VPN on any "
         "network other than a trusted home network."]]),

    "03_travel_expense_policy.pdf": ("Travel and Expense Policy", [
        ["1. Booking travel",
         "All business travel must be booked through the company travel desk, TravelHub, at least 14 days before "
         "departure where possible. Travel needs approval from the reporting manager before booking.",
         "2. Air travel",
         "Economy class is the standard for all flights with a scheduled duration of up to 6 hours. For flights "
         "longer than 6 hours, employees at level L5 and above may book business class; employees below L5 may "
         "book premium economy. Upgrades paid for personally are allowed.",
         "3. Accommodation",
         "Hotel stays are capped at INR 7,000 per night in metro cities (Mumbai, Delhi NCR, Bengaluru, Chennai, "
         "Hyderabad, Kolkata) and INR 4,500 per night in other Indian cities. International hotels are capped at "
         "USD 180 per night unless pre-approved."],
        ["4. Daily allowance (per diem)",
         "The daily allowance for meals and local incidentals is INR 2,500 per day for domestic travel and USD 75 "
         "per day for international travel. Per diem is not paid for days on which meals are provided by a client "
         "or a conference.",
         "5. Submitting expenses",
         "Expense claims must be submitted in PeopleDesk within 30 days of returning from a trip. Itemised "
         "receipts are required for every expense above INR 500. Claims submitted after 60 days will not be "
         "reimbursed.",
         "6. Non-reimbursable items",
         "The company does not reimburse alcohol, traffic or parking fines, personal entertainment, minibar "
         "charges, or travel insurance bought separately (all business travellers are covered by the corporate "
         "travel insurance policy)."]]),

    "04_it_security_policy.pdf": ("IT and Information Security Policy", [
        ["1. Accounts and passwords",
         "Every employee receives a single company identity used for all systems. Passwords must be at least 14 "
         "characters long. Multi-factor authentication (MFA) is mandatory for email, VPN and all cloud tools. "
         "Passwords must never be shared, including with IT staff.",
         "2. Devices",
         "Only company-issued laptops may be used to access customer data. Laptops are encrypted with full-disk "
         "encryption and must be locked when unattended. Personal USB storage devices are blocked on all company "
         "laptops.",
         "3. Networks",
         "The company VPN must be used on any network other than the office network or a trusted home network, "
         "for example in hotels, airports and cafes."],
        ["4. Reporting security incidents",
         "Any suspected security incident, such as a lost laptop, a phishing email that was clicked, or data sent "
         "to the wrong person, must be reported within 1 hour of discovery to the Security Operations team at "
         "security@kestrel.example or on the internal hotline, extension 4444. Employees must not try to "
         "investigate incidents themselves.",
         "5. Data classification",
         "Data is classified as Public, Internal, Confidential or Restricted. Customer data is always Restricted "
         "and may only be stored in approved company systems. Restricted data must never be pasted into public "
         "AI tools.",
         "6. Software",
         "Only software from the company's approved catalogue may be installed. Requests for new software are "
         "raised through the IT service desk and reviewed by the Security team within 5 working days."]]),

    "05_code_of_conduct.pdf": ("Code of Conduct", [
        ["1. Our principles",
         "Employees are expected to act with integrity, treat colleagues and customers with respect, and comply "
         "with all applicable laws. Managers have an additional duty to model these behaviours.",
         "2. Gifts and hospitality",
         "Employees may accept gifts from business partners only if the value is INR 3,000 or less and the gift "
         "is not cash or a cash equivalent. Any gift above INR 3,000 must be declined or declared to the "
         "Compliance team within 7 days.",
         "3. Conflicts of interest",
         "Employees must disclose any actual or potential conflict of interest, such as a second job, a family "
         "member working for a supplier, or an investment in a competitor, to their manager and to Compliance "
         "through the annual declaration and whenever the situation changes."],
        ["4. Anti-harassment (POSH)",
         "The company has zero tolerance for sexual harassment. An Internal Committee (IC) is constituted under "
         "the POSH Act to receive and inquire into complaints. A complaint should be made to the IC within 3 "
         "months of the incident; the IC aims to complete its inquiry within 90 days.",
         "5. Speaking up",
         "Concerns about unethical behaviour can be raised with a manager, HR, or anonymously through the Ethics "
         "Hotline. Retaliation against anyone who raises a concern in good faith is a serious disciplinary "
         "offence.",
         "6. Consequences",
         "Breaches of this Code may lead to disciplinary action, up to and including termination of employment."]]),

    "06_performance_promotion_policy.pdf": ("Performance and Promotion Policy", [
        ["1. Review cycles",
         "Performance is reviewed twice a year, in April and October. Each review includes a self-assessment, "
         "manager assessment and, for levels L4 and above, feedback from at least three peers.",
         "2. Rating scale",
         "Employees are rated on a 5-point scale: 1 Does not meet expectations, 2 Partially meets, 3 Meets, "
         "4 Exceeds, 5 Outstanding. Ratings are calibrated across each department before they are shared.",
         "3. Career levels",
         "The career framework has levels L1 (entry) to L8 (principal / director). Each level has a published "
         "description of expected scope and impact."],
        ["4. Promotion eligibility",
         "To be eligible for promotion, an employee must have spent at least 18 months in their current level and "
         "must have received a rating of 4 or higher in each of the last two review cycles. Promotions take effect "
         "from 1 May or 1 November, following the review cycle.",
         "5. Performance improvement plan",
         "An employee rated 1 in a cycle, or rated 2 in two consecutive cycles, is placed on a Performance "
         "Improvement Plan (PIP) of 60 days with clear, measurable goals and fortnightly check-ins.",
         "6. Salary review",
         "Annual salary revisions are effective from 1 April and are linked to the April review rating and market "
         "benchmarks."]]),

    "07_learning_development_policy.pdf": ("Learning and Development Policy", [
        ["1. Learning budget",
         "Every permanent employee has an annual learning budget of INR 50,000 per financial year for courses, "
         "certifications, conferences and books. Unused budget does not roll over to the next year.",
         "2. Learning time",
         "Employees may use up to 5 working days per year as learning days for approved training, in addition to "
         "annual leave."],
        ["3. Certification reimbursement",
         "Professional certification exam fees are reimbursed in full on passing the exam, for certifications "
         "listed in the approved catalogue (for example cloud, data and security certifications). Failed attempts "
         "are reimbursed once.",
         "4. Clawback",
         "If an employee resigns within 12 months of a single learning expense above INR 25,000 being reimbursed, "
         "100% of that amount is recovered from the final settlement. If they resign between 12 and 24 months, 50% "
         "is recovered.",
         "5. Higher education sponsorship",
         "Employees with at least 2 years of service may apply for sponsorship of a part-time degree, up to "
         "INR 2,00,000 per year, subject to a 2-year service commitment after completion."]]),
}


def build():
    OUT.mkdir(exist_ok=True)
    ss = getSampleStyleSheet()
    for fname, (title, pages) in DOCS.items():
        story = [Paragraph(title, ss["Title"]), Paragraph(COMPANY, ss["Italic"]), Spacer(1, 12)]
        for i, page in enumerate(pages):
            for para in page:
                # short lines that start with a number are section headings
                style = ss["Heading3"] if para[:2].rstrip(".").isdigit() and len(para) < 60 else ss["BodyText"]
                story.append(Paragraph(para, style))
            if i < len(pages) - 1:
                story.append(PageBreak())      # each list item = one PDF page

        def footer(c, d, t=title):
            c.setFont("Helvetica", 8)
            c.drawString(2 * cm, 1.2 * cm, f"{t} - Kestrel Data Labs (fictional) - Page {d.page}")
        SimpleDocTemplate(str(OUT / fname), pagesize=A4, title=title, author="Kestrel Data Labs (fictional)") \
            .build(story, onFirstPage=footer, onLaterPages=footer)
        print("wrote", OUT / fname, f"({len(pages)} pages)")


if __name__ == "__main__":
    build()
