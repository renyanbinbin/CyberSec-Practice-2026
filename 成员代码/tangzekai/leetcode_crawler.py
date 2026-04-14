"""
LeetCode Daily Question Crawler
Crawls the description of LeetCode's daily question from leetcode.cn
"""

import requests
import json
from datetime import datetime
import re
from html import unescape


class LeetCodeCrawler:
    def __init__(self):
        self.base_url = "https://leetcode.cn"
        self.graphql_url = f"{self.base_url}/graphql"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Content-Type": "application/json",
            "Referer": "https://leetcode.cn/",
        })

    def get_daily_question(self):
        """Get today's daily question info"""
        query = """
        query questionOfToday {
            todayRecord {
                date
                userStatus
                question {
                    questionId
                    frontendQuestionId: questionFrontendId
                    difficulty
                    title
                    titleSlug
                    paidOnly: isPaidOnly
                    acRate
                    status
                    topicTags {
                        name
                        nameTranslated: translatedName
                        id
                    }
                }
            }
        }
        """

        try:
            response = self.session.post(
                self.graphql_url,
                json={"query": query},
                timeout=10
            )
            print(f"Daily question API response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                if data.get("data") and data["data"].get("todayRecord"):
                    records = data["data"]["todayRecord"]
                    if records:
                        return records[0]
                else:
                    print(f"API response: {json.dumps(data, indent=2, ensure_ascii=False)[:500]}")
            else:
                print(f"API error: {response.text[:500]}")
        except Exception as e:
            print(f"Request error: {e}")
        return None

    def get_question_detail(self, title_slug):
        """Get question detail by title slug"""
        query = """
        query questionData($titleSlug: String!) {
            question(titleSlug: $titleSlug) {
                questionId
                questionFrontendId
                title
                titleSlug
                translatedTitle
                translatedContent
                content
                difficulty
                topicTags {
                    name
                    translatedName
                }
                hints
                sampleTestCase
            }
        }
        """

        try:
            response = self.session.post(
                self.graphql_url,
                json={
                    "query": query,
                    "variables": {"titleSlug": title_slug}
                },
                timeout=10
            )
            print(f"Question detail API response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                if data.get("data") and data["data"].get("question"):
                    return data["data"]["question"]
                else:
                    print(f"API response: {json.dumps(data, indent=2, ensure_ascii=False)[:500]}")
            else:
                print(f"API error: {response.text[:500]}")
        except Exception as e:
            print(f"Request error: {e}")
        return None

    def clean_html(self, html_content):
        """Clean HTML content to readable text"""
        if not html_content:
            return ""

        # Remove HTML tags but keep content
        text = re.sub(r'<pre[^>]*>.*?</pre>', lambda m: m.group(0).replace('<', '[').replace('>', ']'), html_content, flags=re.DOTALL)
        text = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', text)
        text = re.sub(r'<strong[^>]*>(.*?)</strong>', r'**\1**', text)
        text = re.sub(r'<em[^>]*>(.*?)</em>', r'*\1*', text)
        text = re.sub(r'<li[^>]*>', '• ', text)
        text = re.sub(r'</li>', '\n', text)
        text = re.sub(r'<p[^>]*>', '\n', text)
        text = re.sub(r'</p>', '\n', text)
        text = re.sub(r'<br\s*/?>', '\n', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = unescape(text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def crawl_daily_question(self):
        """Main method to crawl daily question"""
        print("=" * 60)
        print(f"LeetCode Daily Question Crawler")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        print("=" * 60)

        # Get daily question info
        print("\nFetching daily question...")
        daily_info = self.get_daily_question()

        if not daily_info:
            print("Failed to get daily question info")
            return None

        question_info = daily_info.get("question", {})
        title_slug = question_info.get("titleSlug")

        if not title_slug:
            print("Failed to get question slug")
            return None

        # Get question detail
        print(f"Fetching question detail for: {title_slug}")
        detail = self.get_question_detail(title_slug)

        if not detail:
            print("Failed to get question detail")
            return None

        # Build result
        result = {
            "date": daily_info.get("date"),
            "question_id": detail.get("questionFrontendId"),
            "title": detail.get("title"),
            "title_cn": detail.get("translatedTitle"),
            "difficulty": detail.get("difficulty"),
            "url": f"{self.base_url}/problems/{title_slug}/description/",
            "tags": [tag.get("translatedName") or tag.get("name") for tag in detail.get("topicTags", [])],
            "content_cn": self.clean_html(detail.get("translatedContent")),
            "content_en": self.clean_html(detail.get("content")),
            "hints": detail.get("hints", []),
            "sample_test_case": detail.get("sampleTestCase"),
        }

        return result

    def print_result(self, result):
        """Print crawl result in readable format"""
        if not result:
            return

        print("\n" + "=" * 60)
        print(f"Question #{result['question_id']}: {result['title_cn']}")
        print(f"English Title: {result['title']}")
        print(f"Difficulty: {result['difficulty']}")
        print(f"Date: {result['date']}")
        print(f"URL: {result['url']}")
        print(f"Tags: {', '.join(result['tags'])}")
        print("=" * 60)

        print("\n[Description (Chinese)]")
        print("-" * 40)
        print(result['content_cn'])

        if result.get('hints'):
            print("\n[Hints]")
            print("-" * 40)
            for i, hint in enumerate(result['hints'], 1):
                print(f"{i}. {self.clean_html(hint)}")

        print("\n" + "=" * 60)


def main():
    crawler = LeetCodeCrawler()
    result = crawler.crawl_daily_question()

    if result:
        crawler.print_result(result)

        # Print JSON result to terminal
        print("\n" + "=" * 60)
        print("[JSON Result]")
        print("=" * 60)
        print(json.dumps(result, ensure_ascii=False, indent=2))

        # Save to file
        output_file = f"daily_question_{result['date']}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("\n" + "=" * 60)
        print(f"Result saved to: {output_file}")
    else:
        print("Failed to crawl daily question")


if __name__ == "__main__":
    main()
