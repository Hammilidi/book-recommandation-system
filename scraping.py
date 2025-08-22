from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

def scrape_all_books():
    options = Options()
    options.add_argument("--headless")  
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    # Indique le chemin vers msedgedriver.exe
    service = Service(executable_path="./msedgedriver.exe")
    driver = webdriver.Edge(service=service, options=options)
    print("EdgeDriver initialisé")

    wait = WebDriverWait(driver, 15)
    books_data = []

    for page_num in range(1, 51):
        url = f"https://books.toscrape.com/catalogue/page-{page_num}.html"
        
        try:
            print(f"Page {page_num}/50 - Accès à {url}")
            driver.get(url)
            wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".product_pod")))
        except:
            print(f"Page {page_num} non accessible - fin du scraping")
            break

        books = driver.find_elements(By.CSS_SELECTOR, ".product_pod")
        book_urls = []

        for book in books:
            try:
                book_link = book.find_element(By.TAG_NAME, "h3").find_element(By.TAG_NAME, "a")
                book_url = book_link.get_attribute("href")
                book_urls.append(book_url)
            except:
                continue

        print(f"  {len(book_urls)} livres trouvés sur cette page")

        for i, book_url in enumerate(book_urls):
            try:
                driver.get(book_url)
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))

                title = driver.find_element(By.TAG_NAME, "h1").text
                price = driver.find_element(By.CSS_SELECTOR, ".price_color").text
                availability = driver.find_element(By.CSS_SELECTOR, ".availability").text.strip()

                try:
                    description = driver.find_element(By.CSS_SELECTOR, "#product_description ~ p").text
                except:
                    description = "Pas de description"

                try:
                    rating_class = driver.find_element(By.CSS_SELECTOR, ".star-rating").get_attribute("class")
                    rating = rating_class.split()[-1]
                except:
                    rating = "Pas de note"

                try:
                    img_url = driver.find_element(By.CSS_SELECTOR, ".item.active img").get_attribute("src")
                except:
                    img_url = "Pas d'image"

                books_data.append({
                    "Titre": title,
                    "Prix": price,
                    "Disponibilité": availability,
                    "Description": description,
                    "Note": rating,
                    "Image": img_url,
                    "Page": page_num
                })

                if (i + 1) % 5 == 0:
                    print(f"  {i+1}/{len(book_urls)} livres collectés (Total: {len(books_data)})")

                time.sleep(0.02)

            except Exception as e:
                print(f"  Erreur livre {i+1}: {e}")
                continue

        print(f"Page {page_num} terminée - Total: {len(books_data)} livres")
        time.sleep(0.5)

    driver.quit()

    if books_data:
        df = pd.DataFrame(books_data)
        filename = f"tous_les_livres_{len(books_data)}.csv"
        df.to_csv(filename, index=False, encoding='utf-8')

        print("\nSCRAPING TERMINE")
        print(f"Total de livres: {len(books_data)}")
        print(f"Fichier: {filename}")

        pages_scrapped = df['Page'].nunique()
        print(f"Pages parcourues: {pages_scrapped}")
        print(f"Moyenne par page: {len(books_data)/pages_scrapped:.1f}")

        print("\nQuelques exemples:")
        for i, book in enumerate(books_data[:5]):
            print(f"  {i+1}. {book['Titre']} - {book['Prix']}")
    else:
        print("Aucun livre collecté")

if __name__ == "__main__":
    scrape_all_books()
