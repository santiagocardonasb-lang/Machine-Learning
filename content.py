"""Text content for the Machine Learning use case pages."""

USE_CASES = {
    1: {
        "name": "Credit card fraud detection",
        "context": (
            "A bank processes millions of card transactions every day. A small "
            "fraction of them are fraudulent, and reviewing each one manually is "
            "impossible. Fraud has to be caught in the seconds before the payment "
            "is approved."
        ),
        "data": (
            "Transaction history: amount, time of day, merchant category, country, "
            "distance from the cardholder's usual locations, and the number of "
            "transactions in the previous hour. Each historical transaction is "
            "labelled as legitimate or fraudulent."
        ),
        "goal": (
            "Classify every incoming transaction as legitimate or fraudulent, and "
            "return a risk score the bank can use to block it or ask for a second "
            "confirmation."
        ),
        "ml_type": "Supervised learning - binary classification",
        "benefit": (
            "Cuts financial losses and protects the customer without freezing "
            "legitimate purchases. The model reviews every transaction in "
            "milliseconds, which no human team could do."
        ),
    },
    2: {
        "name": "Customer segmentation in retail",
        "context": (
            "A supermarket chain sends the same promotions to its entire customer "
            "base. The response rate is low because the offers rarely match what "
            "each customer actually buys, and the marketing budget is wasted."
        ),
        "data": (
            "Loyalty card records: purchase frequency, average basket value, "
            "product categories bought, preferred store, time of day, and response "
            "to previous campaigns. There are no labels: nobody has defined the "
            "customer groups in advance."
        ),
        "goal": (
            "Discover natural groups of customers with similar behaviour, so each "
            "group can receive the promotions that fit it."
        ),
        "ml_type": "Unsupervised learning - clustering",
        "benefit": (
            "Higher campaign response rates with the same budget, and a clearer "
            "picture of who the customers actually are, beyond age and address."
        ),
    },
    3: {
        "name": "Predictive maintenance of industrial machinery",
        "context": (
            "A manufacturing plant repairs its machines only after they break. Each "
            "unplanned stop halts the production line for hours and costs far more "
            "than a scheduled repair would have."
        ),
        "data": (
            "Sensor readings collected continuously: vibration, temperature, "
            "pressure, electrical consumption and running hours, together with the "
            "maintenance log recording when each machine actually failed."
        ),
        "goal": (
            "Predict how many hours or days remain before a machine fails, so "
            "maintenance can be scheduled before the breakdown happens."
        ),
        "ml_type": "Supervised learning - regression",
        "benefit": (
            "Fewer unplanned stops, longer equipment life, and maintenance staff "
            "scheduled when the plant is not producing instead of in an emergency."
        ),
    },
    4: {
        "name": "Diagnostic support from medical images",
        "context": (
            "A radiology department receives more scans than its specialists can "
            "review promptly. Delays in reading an image can postpone a diagnosis "
            "at the stage where early treatment matters most."
        ),
        "data": (
            "Thousands of X-rays, CT scans and MRIs already reviewed and labelled "
            "by radiologists, indicating whether a lesion is present and where "
            "it is located in the image."
        ),
        "goal": (
            "Classify each new image by the presence or absence of findings and "
            "highlight the regions that need attention, so the specialist reviews "
            "the most urgent cases first."
        ),
        "ml_type": "Supervised learning - image classification",
        "benefit": (
            "Shorter waiting times and a second opinion that never gets tired. The "
            "model does not replace the radiologist: it prioritizes the queue and "
            "flags what could be missed."
        ),
    },
}