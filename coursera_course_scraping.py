## Importing libraries
"""## Define scraping funtion"""

def auto_Scrapper_Class(html_tag,course_case,tag_class, div_class=None):
    """
    The function auto_Scrapper_Class is used to get three parameters that is the tag,what to scrap and get the content scrapped and class it belongs.
    """
    for i in range(1,50): # adjust as needed, according to current coursera website, there are 83 pages for all courses
        url = "https://www.coursera.org/courses?query=cybersecurity&page=" +str(i)

        #Use below url to gain more customization on the result with different query
        #url = "https://www.coursera.org/search?query=data%20science&page=" +str(i)

        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')

        if div_class:
            elements = soup.find_all('div',  class_ = div_class)
            # Ensure exactly 12 elements are processed for each page
            while len(elements) < 12:
                elements.append(None) # Pad with None if less than 12

            for name in elements[:12]: # Process only the first 12 elements
                if name:
                    x = name.get_text()
                    course_case.append(x)
                else:
                    course_case.append(None)

        else:
            element = soup.find_all(html_tag,  class_ = tag_class)
            # Ensure exactly 12 elements are processed for each page
            while len(element) < 12:
                element.append(None) # Pad with None if less than 12

            for name in element[:12]: # Process only the first 12 elements
                x = name.get_text() if name else None
                if x:
                    course_case.append(x)
                else:
                    course_case.append(None) # Append None for missing or empty elements

course_title = []
course_organization = []
course_Certificate_type = []
course_rating = []
course_difficulty = []
course_review_counts = []
course_skills = []

#scrap the course title
auto_Scrapper_Class('h3',course_title, tag_class='cds-CommonCard-title css-6ecy9b')

#scrap the other information as per coursera's website html
auto_Scrapper_Class('p',course_organization,'cds-ProductCard-partnerNames css-vac8rf')
#auto_Scrapper_Class('div',course_Certificate_type,'_jen3vs _1d8rgfy3')
auto_Scrapper_Class('p',course_rating,'cds-RatingStat-meter')
auto_Scrapper_Class('p',course_difficulty,'cds-119 cds-Typography-base css-dmxkm1 cds-121', 'cds-CommonCard-metadata')
auto_Scrapper_Class('p',course_review_counts,'cds-119 cds-Typography-base css-dmxkm1 cds-121', 'product-reviews css-pn23ng')
auto_Scrapper_Class('p',course_skills,'cds-119 cds-Typography-base css-dmxkm1 cds-121', 'cds-CommonCard-bodyContent' )

"""## Clean the scraped data"""

data = {
    'Title': course_title,
    'Organization': course_organization,
    'Skills': course_skills,
    'Ratings':course_rating,
    'Review counts':course_review_counts,
    'Metadata': course_difficulty
}

# Check the lengths of the lists
print(f"Length of course_title: {len(course_title)}")
print(f"Length of course_organization: {len(course_organization)}")
print(f"Length of course_skills: {len(course_skills)}")
print(f"Length of course_rating: {len(course_rating)}")
print(f"Length of course_review_counts: {len(course_review_counts)}")
print(f"Length of course_difficulty: {len(course_difficulty)}")


df = pd.DataFrame(data)
df['Skills'] = df['Skills'].str.replace("Skills you'll gain:", '', regex=False)
df

df.to_csv("coursera_cyber_security.csv")

"""## Extras:
#### In my case, I only want the name of skills that can be obtained from courses in Coursera
"""

skills_column = df['Skills']

skills_column = [str(skill) if skill is not None else '' for skill in skills_column]

# Concatenate all skills into a single string
all_skills_text = ', '.join(skills_column)

# Split the string into a list of skills
all_skills_list = [skill.strip() for skill in all_skills_text.split(',')]

# Get unique skills
distinct_skills = list(set(all_skills_list))

distinct_skills_df = pd.DataFrame({'Distinct Skills': distinct_skills,'source':"coursera"})
len(distinct_skills)

distinct_skills_df

distinct_skills_df.to_csv("skills_coursera.csv")
